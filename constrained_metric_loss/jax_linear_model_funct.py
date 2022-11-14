import jax.numpy as jnp
import jax.nn as jnn
from jax import jit

# from jax import lax
# from functools import partial

import optax

from jax.example_libraries import optimizers
import jax


def logsigmoid_approx(x, y):
    return -jnn.log_sigmoid((2 * y - 1) * x)


def leaky_relu_approx(x, y, negative_slope):
    return jnn.relu(1 - jnn.leaky_relu((2 * y - 1) * x), negative_slope=negative_slope)


def apply_model(x, params):
    if len(params) > 1:
        hidden_layers = params[:-1]
        for weights, biases in hidden_layers:
            x = jax.nn.relu(jnp.dot(x, weights.T) + biases)

    w_last, b_last = params[-1]
    return jnp.dot(x, w_last.T) + b_last


@jit
def model_loss(beta, x, y, min_prec, lmbda, lmbda2):
    f = apply_model(x, beta)

    tpc = jnp.dot(y.flatten(), 1 - logsigmoid_approx(f, 1.0).flatten())
    fpc = jnp.dot(1 - y.flatten(), logsigmoid_approx(f, 0.0).flatten())

    Nplus = jnp.sum(y)

    g = -tpc + min_prec / (1.0 - min_prec) * fpc + Nplus

    loss = -tpc + lmbda * jnn.relu(g) + lmbda2 * jnn.relu(-tpc - fpc)

    return loss


opt_init, opt_update, get_params = optimizers.adam(0.01)


@jit
def step(step, opt_state, x, y, min_prec, lmbda, lmbda2):
    value, grads = jax.value_and_grad(model_loss, argnums=0)(get_params(opt_state), x, y, min_prec, lmbda, lmbda2)
    opt_state = opt_update(step, grads, opt_state)
    return value, opt_state


def fit_model(beta_init, x, y, min_prec, lmbda, lmbda2, n_epochs=1000, batch_size=8):
    opt_state = opt_init(beta_init)
    n_batches = len(y) // batch_size + 1
    for i in range(n_epochs):
        for j in range(n_batches - 1):
            x_batch = x[j * batch_size : (j + 1) * batch_size]
            y_batch = y[j * batch_size : (j + 1) * batch_size]
            value, opt_state = step(i * n_batches + j, opt_state, x_batch, y_batch, min_prec, lmbda, lmbda2)

    return value, get_params(opt_state)


def fit_model_optax(
    params: optax.Params, optimizer: optax.GradientTransformation, x, y, min_prec, lmbda, lmbda2, n_epochs=1000
) -> optax.Params:
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, x, y, min_prec, lmbda, lmbda2):
        loss_value, grads = jax.value_and_grad(model_loss, argnums=0)(params, x, y, min_prec, lmbda, lmbda2)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    for i in range(n_epochs):
        params, opt_state, loss_value = step(params, opt_state, x, y, min_prec, lmbda, lmbda2)

    return loss_value, params
