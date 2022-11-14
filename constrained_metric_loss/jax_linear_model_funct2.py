import jax.numpy as jnp
import jax.nn as jnn
from jax import jit
from functools import partial
from constrained_metric_loss.sigmoid_optimizer import get_sigmoid_params_bfgs

from jax.example_libraries import optimizers
import jax


def logsigmoid_approx():
    return lambda x, y: -jnn.log_sigmoid((2 * y - 1) * x)


def leaky_relu_approx(negative_slope):
    return lambda x, y: jnn.relu(1 - jnn.leaky_relu((2 * y - 1) * x, negative_slope=negative_slope))


def semi_rath_hughes_approx(gamma, delta, eps):
    mhat, bhat = get_sigmoid_params_bfgs(gamma, delta, eps, "upper")
    return lambda x, y: (1 + gamma * delta) * jnn.sigmoid(-(2 * y - 1) * mhat * x + bhat)


def apply_model(x, params):
    if len(params) > 1:
        hidden_layers = params[:-1]
        for weights, biases in hidden_layers:
            x = jax.nn.relu(jnp.dot(x, weights.T) + biases)

    w_last, b_last = params[-1]
    return jnp.dot(x, w_last.T) + b_last


def eban_model_loss(beta, x, y, min_prec, lmbda, lmbda2, approx_func):
    f = apply_model(x, beta)

    tpc = jnp.dot(y.flatten(), 1 - approx_func(f, 1.0).flatten())
    fpc = jnp.dot(1 - y.flatten(), approx_func(f, 0.0).flatten())

    Nplus = jnp.sum(y)

    g = -tpc + min_prec / (1.0 - min_prec) * fpc + Nplus

    loss = -tpc + lmbda * jnn.relu(g) + lmbda2 * jnn.relu(-tpc - fpc)

    return loss


def create_eban_loss(min_prec, lmbda, lmbda2, approx_func):
    return lambda beta, x, y: eban_model_loss(beta, x, y, min_prec, lmbda, lmbda2, approx_func)


def rath_hughes_loss(beta, x, y, min_prec, lmbda, gamma, delta, eps):
    mhat, bhat = get_sigmoid_params_bfgs(gamma, delta, eps, "upper")
    mtilde, btilde = get_sigmoid_params_bfgs(gamma, delta, eps, "lower")
    f = apply_model(x, beta)

    tpc = jnp.dot(y.flatten(), (1 + gamma * delta) * jnn.sigmoid(mtilde * f + btilde))
    fpc = jnp.dot(1 - y.flatten(), (1 + gamma * delta) * jnn.sigmoid(mhat * f + bhat))

    Nplus = jnp.sum(y)

    g = -tpc + (min_prec / (1.0 - min_prec)) * fpc + gamma * delta * Nplus

    loss = -tpc + lmbda * jnn.relu(g)

    return loss


def create_rath_hughes_loss(min_prec, lmbda, gamma, delta, eps):
    return lambda beta, x, y: rath_hughes_loss(beta, x, y, min_prec, lmbda, gamma, delta, eps)


opt_init, opt_update, get_params = optimizers.adam(0.01)


@partial(jit, static_argnums=(0,))
def step(loss, step, opt_state, x, y):
    value, grads = jax.value_and_grad(loss, argnums=0)(get_params(opt_state), x, y)
    opt_state = opt_update(step, grads, opt_state)
    return value, opt_state


def fit_model2(loss, beta_init, x, y, n_epochs=1000, batch_size=1024):
    opt_state = opt_init(beta_init)
    n_batches = len(y) // batch_size + 1
    for i in range(n_epochs):
        for j in range(n_batches - 1):
            x_batch = x[j * batch_size : (j + 1) * batch_size]
            y_batch = y[j * batch_size : (j + 1) * batch_size]
            value, opt_state = step(loss, i * n_batches + j, opt_state, x_batch, y_batch)

    return value, get_params(opt_state)


def rath_hughes_loss3(beta, x, y, min_prec, lmbda, gamma, delta, mhat, bhat, mtilde, btilde):
    f = apply_model(x, beta)

    tpc = jnp.dot(y.flatten(), (1 + gamma * delta) * jnn.sigmoid(mtilde * f + btilde))
    fpc = jnp.dot(1 - y.flatten(), (1 + gamma * delta) * jnn.sigmoid(mhat * f + bhat))

    Nplus = jnp.sum(y)

    g = -tpc + (min_prec / (1.0 - min_prec)) * fpc + gamma * delta * Nplus

    loss = -tpc + lmbda * jnn.relu(g)

    return loss


@jit
def step3(step, opt_state, x, y, min_prec, lmbda, gamma, delta, mhat, bhat, mtilde, btilde):
    value, grads = jax.value_and_grad(rath_hughes_loss3, argnums=0)(
        get_params(opt_state), x, y, min_prec, lmbda, gamma, delta, mhat, bhat, mtilde, btilde
    )
    opt_state = opt_update(step, grads, opt_state)
    return value, opt_state


def fit_model3(beta_init, x, y, min_prec, lmbda, gamma, delta, eps, n_epochs=1000, batch_size=1024):
    opt_state = opt_init(beta_init)
    n_batches = len(y) // batch_size + 1
    mhat, bhat = get_sigmoid_params_bfgs(gamma, delta, eps, "upper")
    mtilde, btilde = get_sigmoid_params_bfgs(gamma, delta, eps, "lower")
    key = jax.random.PRNGKey(42)
    for i in range(n_epochs):
        if i % 100 == 0:
            x = jax.random.permutation(key, x, axis=0, independent=False)
            y = jax.random.permutation(key, y, axis=0, independent=False)
        for j in range(n_batches - 1):  # range(1):  #
            x_batch = x[j * batch_size : (j + 1) * batch_size]
            y_batch = y[j * batch_size : (j + 1) * batch_size]
            value, opt_state = step3(
                i * n_batches + j,
                opt_state,
                x_batch,
                y_batch,
                min_prec,
                lmbda,
                gamma,
                delta,
                mhat,
                bhat,
                mtilde,
                btilde,
            )

    return value, get_params(opt_state)
