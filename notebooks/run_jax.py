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

def logit(x):
    return np.log(x/(1-x))


# +
from data import toydata

x, y, _, _, _, _ = toydata.create_toy_dataset()
# x, y, _, _, _, _ = toydata.create_toy_dataset_large()

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
from sklearn.linear_model import LogisticRegression, SGDClassifier

sklearnlogreg = LogisticRegression(C=1)
sklearnlogreg = sklearnlogreg.fit(
    data_dict["X_train"], data_dict["y_train"]
)

sk_param_init = [[sklearnlogreg.coef_.flatten(), sklearnlogreg.intercept_]]
sk_param_init


# -


plt.scatter(
    data_dict["X_train"][:, 0], 
    data_dict["X_train"][:, 1],
    c=data_dict["y_train"]
)
x_plt = np.linspace(-3, 2, 10)
plt.plot(x_plt, - (1 / sk_param_init[0][0][1]) * (sk_param_init[0][0][0] * x_plt + sk_param_init[0][1][0] + 0.0))

# +
# param_init = [[np.array([-1.1103404, -0.59285593]), np.array([-3.7731001])]]

# param_init = [[np.array([-4.4770846, -3.303846]),np.array([ -5.0092807])]]

# param_init = [[np.array([-11.315149,  -8.30925 ]),np.array([ -12.072914])]]

# param_init = [[np.array([-17.01511  , -12.5248165]),np.array([ -18.090153])]]

# param_init = [[np.array([-0.0, -0.0]), np.array([0.0])]]

param_init = [[np.array([-2.7280293, -0.7228309]), np.array([3.4114962])]]



# +
# %%time


# loss = create_eban_loss(0.8, 1000, 2000, semi_rath_hughes_approx(gamma=7, delta=0.035, eps=0.75))
# loss = create_eban_loss(0.8, 1_000, 2_000, logsigmoid_approx())
# loss = create_eban_loss(0.8, 1000, 2000, leaky_relu_approx(0.1))
loss = create_rath_hughes_loss(0.8, 1000, gamma=7, delta=0.035, eps=0.75)

loss, parameters = fit_model2(
    loss,
    param_init,
    data_dict["X_train"],
    data_dict["y_train"],
    n_epochs=1_000
)

# +
from sklearn.metrics import precision_score, recall_score, precision_recall_curve

y_hat = jnn.sigmoid(apply_model(data_dict["X_test"], parameters))
y_pred = (y_hat > 0.5).astype(int)


precision, recall, thresholds = precision_recall_curve(data_dict["y_test"], y_hat)


plt.scatter(recall, precision)
precision_score(data_dict["y_test"], y_pred), recall_score(data_dict["y_test"], y_pred), thresholds[1]

# +

plt.scatter(
    data_dict["X_train"][:, 0], 
    data_dict["X_train"][:, 1],
    c=data_dict["y_train"]
)
x_plt = np.linspace(-1.5, 2, 10)
plt.plot(x_plt, - (1 / parameters[0][0][1]) * (parameters[0][0][0] * x_plt + parameters[0][1][0]))

parameters

# +
sklearn_yhat = sklearnlogreg.predict_proba(data_dict["X_test"])[:, 1]

precision, recall, thresholds = precision_recall_curve(data_dict["y_test"], sklearn_yhat)


plt.plot(recall, precision)

logit(thresholds[precision[1:-5].argmax()])
logit(0.8)
# -



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


# +
mlp_params_init = MLP_init(layer_widths = [2, 4, 1], prng_key=jax.random.PRNGKey(2))


mlp_params_init = [[jnp.array([[ 6.4603396e-02, -4.0374018e-04],
               [-5.7415307e-01, -1.8430549e-01],
               [-5.0981516e-01,  9.9774472e-02],
               [-5.7155782e-01, -2.5568867e-01]]),
  jnp.array([-0.04057087,  0.58211887,  0.55022043,  0.5816231 ])],
 [jnp.array([[0.06169445, 0.5239211 , 0.7071111 , 0.5371694 ]]),
  jnp.array([0.5464785])]]

# +
# %%time


# loss = create_eban_loss(0.8, 1000, 2000, semi_rath_hughes_approx(gamma=7, delta=0.035, eps=0.75))
# loss = create_eban_loss(0.8, 1_000, 2_000, logsigmoid_approx())
# loss = create_eban_loss(0.8, 1000, 2000, leaky_relu_approx(0.1))
loss = create_rath_hughes_loss(0.8, 10_000, gamma=7, delta=0.035, eps=0.75)

loss, parameters = fit_model2(
    loss,
    mlp_params_init,
    data_dict["X_train"],
    data_dict["y_train"],
    n_epochs=10_000
)

# +
from sklearn.metrics import precision_score, recall_score, precision_recall_curve

y_hat = jnn.sigmoid(apply_model(data_dict["X_test"], parameters))
y_pred = (y_hat > 0.5).astype(int)
precision_score(data_dict["y_test"], y_pred), recall_score(data_dict["y_test"], y_pred)

precision, recall, thresholds = precision_recall_curve(data_dict["y_test"], y_hat)


plt.scatter(recall, precision)
precision_score(data_dict["y_test"], y_pred), recall_score(data_dict["y_test"], y_pred), thresholds[1]
# -

parameters

# +
from sklearn.neural_network import MLPClassifier

sklearn_mlp = MLPClassifier(hidden_layer_sizes=(4, ))
sklearn_mlp.fit(data_dict["X_train"], data_dict["y_train"])

# +

sklearn_mlp_yhat = sklearn_mlp.predict_proba(data_dict["X_test"])[:, 1]

precision, recall, thresholds = precision_recall_curve(data_dict["y_test"], sklearn_mlp_yhat)


plt.plot(recall, precision)
# -


