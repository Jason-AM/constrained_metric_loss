# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python [conda env:min_prec]
#     language: python
#     name: conda-env-min_prec-py
# ---

# %load_ext autoreload
# %autoreload 2

# +
import torch
import torch.nn as nn
import jax.numpy as jnp
import jax.nn as jnn
from jax import jit
import matplotlib.pyplot as plt
from timeit import timeit
import numpy as np

import constrained_metric_loss.min_precision_loss as torch_min_prec
import constrained_metric_loss.jax_linear_model as jax_min_prec
# -

# # basic tests

from jax.example_libraries import optimizers
from sklearn.model_selection import train_test_split



# +
from data import toydata

x, y, _, _, _, _ = toydata.create_toy_dataset()

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42, stratify=y
)

data_dict = {
    "X_train": X_train, 
    "X_test": X_test, 
    "y_train": y_train, 
    "y_test": y_test
}

# +
from sklearn.linear_model import LogisticRegression

sklearnlogreg = LogisticRegression()
sklearnlogreg = sklearnlogreg.fit(
    data_dict["X_train"], data_dict["y_train"]
)

param_init = [[sklearnlogreg.coef_.flatten(), sklearnlogreg.intercept_]]

# -

from constrained_metric_loss.jax_linear_model_funct import fit_model


# + tags=[]
# %%time
x = data_dict["X_train"]
value, opt_state = fit_model(
    param_init,
    x,
    data_dict["y_train"],
    0.8, 
    1000, 
    2000,
    n_batches=10_000
)

# -

value, opt_state[0]


# +
# %%time

from constrained_metric_loss.min_precision_loss import (
    MinPrec01LossApprox, ZeroOneApproximation

)
from constrained_metric_loss.linear_model import LinearModel
import torch.optim as optim


zero_one_loss_approximation = ZeroOneApproximation(
    name_of_approx = "logsigmoid_approx",
    params_dict = {}
)


model = LinearModel(
    nfeat=data_dict["X_train"].shape[1],
    model_param_init=np.concatenate(param_init[0]),
    loss=MinPrec01LossApprox,
    loss_arguments={
        "min_prec": 0.8, 
        "lmbda": 1000., 
        'lmbda2': 2000., 
        'zero_one_loss_approx': zero_one_loss_approximation,
    },
)

optimizer = optim.Adam(model.parameters(), lr=0.01)

training_losses = model.fit(
    data_dict["X_train"], 
    data_dict["y_train"].reshape(-1,1),
    optimizer, 
    n_batches=10_000
)


# -

training_losses[-1], list(model.parameters())

# +

import optax
from constrained_metric_loss.jax_linear_model_funct import fit_model_optax


# +
# %%time

x = data_dict["X_train"]

optimizer = optax.adam(learning_rate=1e-2)
params = fit_model_optax(
    param_init, 
    optimizer, 
    x, 
    data_dict["y_train"],
    0.8, 
    1000., 
    2000.,
    n_batches=10_000
)
# -


params

# # non linear

# +
import jax

def MLP_init(layer_widths, prng_key, scale=0.02):
    model = []
    # split of keys to generate random numbers
    keys = jax.random.split(prng_key, num=len(layer_widths)-1)
    
    zipped_layer_size_keys = zip(layer_widths[:-1], layer_widths[1:], keys)
    for input_width, output_width, key in zipped_layer_size_keys:
        weight_key, bias_key = jax.random.split(key)
        
        model.append(
            [
                scale*jax.random.normal(
                    weight_key, shape=(output_width, input_width)
                ),
                scale*jax.random.normal(bias_key, shape=(output_width,))
            ]
        )

    return model


# -

params = MLP_init(layer_widths = [2, 4, 1], prng_key=jax.random.PRNGKey(2))

# +
# from constrained_metric_loss.jax_linear_model_funct import fit_model

# +
# # %%time

# value, opt_state = fit_model(
#     params, 
#     data_dict["X_train"], 
#     data_dict["y_train"],
#     0.8, 
#     1000, 
#     2000
# )

# +
# value, opt_state[0]
# -



# +
from constrained_metric_loss.mlp import MLP
from constrained_metric_loss.min_precision_loss import (
    MinPrec01LossApprox, ZeroOneApproximation

)
import torch.optim as optim

zero_one_loss_approximation = ZeroOneApproximation(
    name_of_approx = "logsigmoid_approx",
    params_dict = {}
)


model = MLP(
    nfeat=data_dict["X_train"].shape[1],
    hidden_width=4, 
    loss=MinPrec01LossApprox,
    loss_arguments={
        "min_prec": 0.8, 
        "lmbda": 1000., 
        'lmbda2': 2000., 
        'zero_one_loss_approx': zero_one_loss_approximation,
    },
)

# -

list(model.parameters())

ls_params = [jnp.array(i.detach().numpy()) for i in list(model.parameters())]
torch_jax_init_params = [[ls_params[0], ls_params[1]], [ls_params[2], ls_params[3]]]

torch_jax_init_params

# +
# %%time

optimizer = optim.Adam(model.parameters(), lr=0.01)

training_losses = model.fit(
    data_dict["X_train"], 
    data_dict["y_train"].reshape(-1,1),
    optimizer, 
    n_batches=2
)
# -

training_losses[-1], list(model.parameters())

# +
# %%time 

loss_value, params = fit_model(
    torch_jax_init_params, 
    data_dict["X_train"], 
    data_dict["y_train"],
    0.8, 
    1000., 
    2000.,
    n_batches=2
)

# +
# %%time
from constrained_metric_loss.jax_linear_model_funct import fit_model_optax
import optax

x = data_dict["X_train"]
optimizer = optax.adam(learning_rate=0.01)
loss_value, params = fit_model_optax(
    torch_jax_init_params, 
    optimizer, 
    x, 
    data_dict["y_train"],
    0.8, 
    1000., 
    2000.,
    n_batches=2
)

# -

loss_value, params




