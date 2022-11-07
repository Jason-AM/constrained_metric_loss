import inspect
import jax.numpy as jnp
import jax.nn as jnn
from jax import jit
from functools import partial
import numpy as np
from jax.example_libraries import optimizers
import jax

from constrained_metric_loss.sigmoid_optimizer import get_sigmoid_params_bfgs


class ZeroOneApproximation:
    def __init__(self, name_of_approx: str, params_dict: dict):

        # ensure name chosen from available options
        try:
            available_approxs = [name for name in dir(self) if "__" not in name]
            assert name_of_approx in available_approxs
            self.zero_one_loss_approx_func = getattr(self, name_of_approx)
        except AssertionError:
            raise ValueError(
                "chosen approximation not in list of possible options. "
                "Please select one of the following: "
                f"{available_approxs}"
            )

        # ensure the params dictionary contains the correct inputs for the chosen name
        try:
            required_params = inspect.getfullargspec(self.zero_one_loss_approx_func).args
            assert all(item in params_dict.keys() for item in required_params)
            self.zero_one_loss_approx_func = self.zero_one_loss_approx_func(**params_dict)
        except AssertionError:
            raise ValueError(
                "Parameters dictionary is missing arguments for chosen loss approximation. "
                f"For your chosen loss {name_of_approx} you need to define: "
                f"{required_params}"
            )

    @staticmethod
    def leaky_relu_approx(negative_slope):
        return lambda x, y: jnn.ReLU()(1 - jnn.LeakyReLU(negative_slope=negative_slope)((2 * y - 1) * x))

    @staticmethod
    def logsigmoid_approx(x, y):
        return lambda x, y: -jnp.log(jnn.sigmoid((2 * y - 1) * x))

    @staticmethod
    def semi_rath_hughes_approx(gamma, delta, eps):
        mhat, bhat = get_sigmoid_params_bfgs(gamma, delta, eps, "upper")
        return lambda x, y: (1 + gamma * delta) * jnn.sigmoid(-(2 * y - 1) * mhat * x + bhat)

    @jit
    def calc_loss(self, x, y):
        return self.zero_one_loss_approx_func(x, y)


class MinPrec01LossApprox:
    def __init__(self, min_prec: float, lmbda: float, lmbda2: float, zero_one_loss_approx: ZeroOneApproximation):
        self.min_prec = min_prec
        self.lmbda = lmbda
        self.lmbda2 = lmbda2
        self.zero_one_loss_approx = zero_one_loss_approx

    @partial(jit, static_argnums=(0,))
    def forward(self, f, y):

        tpc = jnp.dot(y.flatten(), 1 - self.zero_one_loss_approx(f, 1.0).flatten())
        fpc = jnp.dot(1 - y.flatten(), self.zero_one_loss_approx(f, 0.0).flatten())

        Nplus = jnp.sum(y)

        g = -tpc + self.min_prec / (1.0 - self.min_prec) * fpc + Nplus

        loss = -tpc + self.lmbda * jnn.relu(g) + self.lmbda2 * jnn.relu(-tpc - fpc)

        return loss


class LinearModel:
    """Produces classification scores for given loss"""

    def __init__(
        self,
        nfeat,
        model_param_init,
        loss,
        loss_arguments,
    ):

        self.nfeat = nfeat
        self.beta = model_param_init
        assert nfeat + 1 == len(model_param_init), "Using the coefficient trick so need to add one extra parameter"

        self.loss = loss
        self.loss_arguments = loss_arguments

        self.loss_func = self.loss(**self.loss_arguments)

    @partial(jit, static_argnums=(0,))
    def loss(self, beta, x, y):
        x = np.column_stack((x, np.ones([x.shape[0]])))

        f = x @ beta

        return self.loss_func.forward(f, y)

    @partial(jit, static_argnums=(0,))
    def sigmoid_of_score(self, xtest):

        if len(xtest.shape) == 1:
            xtest = xtest.reshape([1, self.nfeat])

        xtest = np.column_stack((xtest, np.ones([xtest.shape[0]])))

        f = xtest @ self.beta
        return jnn.sigmoid(f).flatten()

    #     def fit(self, x, y, optimizer, n_batches=200):
    #         training_loss = []
    #         for _ in range(n_batches):
    #             optimizer.zero_grad()
    #             loss = self.forward(x, y)
    #             loss.backward()
    #             optimizer.step()

    #             training_loss.append(loss.detach().numpy())

    #         return training_loss

    def fit(self, x, y):
        opt_init, opt_update, get_params = optimizers.adam(0.01)
        opt_state = opt_init(self.beta)
        print(self.loss_func.zero_one_loss_approx)

        def step(step, opt_state):
            value, grads = jax.value_and_grad(self.loss, argnums=0)(get_params(opt_state), x, y)
            opt_state = opt_update(step, grads, opt_state)
            return value, opt_state

        for i in range(10_000):
            value, opt_state = step(i, opt_state)
