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

from constrained_metric_loss.jax_linear_model_funct2 import (
    apply_model, 
    fit_model2, 
    create_eban_loss, 
    logsigmoid_approx, 
    leaky_relu_approx, 
    semi_rath_hughes_approx, 
    create_rath_hughes_loss
)

from jax.example_libraries import optimizers
from sklearn.model_selection import train_test_split
# -

# # basic tests

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


# +
# %%time


# loss = create_eban_loss(0.8, 1000, 2000, semi_rath_hughes_approx(gamma=7, delta=0.035, eps=0.75))
loss = create_rath_hughes_loss(0.8, 1000, gamma=7, delta=0.035, eps=0.75)

loss, parameters = fit_model2(
    loss,
    param_init,
    x,
    data_dict["y_train"],
    n_epochs=10_000
)
# -

jnn.sigmoid(apply_model(data_dict["X_test"], parameters))

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




