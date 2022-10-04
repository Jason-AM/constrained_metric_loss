# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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
import numpy as np
import os
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# -

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# +
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objects as go

init_notebook_mode()

# -

from constrained_metric_loss.min_precision_loss import (
    MinPrecLoss, MinPrecLossLogForm, MinPrecLeakyLoss
)
from constrained_metric_loss.bce_type_loss import BCELogitManual

# # data

# Data set from paper found at https://github.com/tufts-ml/false-alarm-control/blob/main/toy_example_comparing_BCE_Hinge_and_Sigmoid.ipynb

# +
from data import toydata

x_toy, y_toy, _, _, _, _ = toydata.create_toy_dataset()

x = torch.from_numpy(x_toy).float()
y = torch.from_numpy(y_toy).float()

# +
# from sklearn.datasets import make_moons

# x_moons, y_moons = make_moons(n_samples = 10000, noise=0.1, random_state=7)

# x = torch.from_numpy(x_moons).float()
# y = torch.from_numpy(y_moons).float()

# +
# n = 10000

# rng = np.random.default_rng(5)

# x1 = rng.normal(0, 1, size=n)
# x2 = rng.normal(3, 2.2, size=n)


# beta0_true = -0.4
# beta1_true = -5.3
# beta2_true = 3.1

# p = 1/(1+np.exp(-(beta0_true + beta1_true*x1 + beta2_true*x2)))

# y = torch.from_numpy(rng.binomial(1, p, size=n)).float()
# x = torch.from_numpy(np.column_stack((x1, x2))).float()
# -

# #### params from sklearn

# +

sklearnlogreg = LogisticRegression()
sklearnlogreg = sklearnlogreg.fit(x,y)
sklearnbetas = np.concatenate([sklearnlogreg.coef_.flatten(), sklearnlogreg.intercept_])
# -

sklearnbetas


# # Losses

def loss_for_bce(beta):
    torch_beta = torch.from_numpy(beta).float()
    model_func = lambda xx:  xx @ torch_beta

    x_w_dummy_for_int = np.column_stack((x, np.ones([x.shape[0]])))
    x_w_dummy_for_int = torch.from_numpy(x_w_dummy_for_int).float()

    loss_funct = nn.BCEWithLogitsLoss(reduction='sum')

    return loss_funct(model_func(x_w_dummy_for_int), y).numpy()



def loss_for_bce_manual(beta):
    torch_beta = torch.from_numpy(beta).float()
    model_func = lambda xx:  xx @ torch_beta

    x_w_dummy_for_int = np.column_stack((x, np.ones([x.shape[0]])))
    x_w_dummy_for_int = torch.from_numpy(x_w_dummy_for_int).float()

    loss_funct = BCELogitManual()

    return loss_funct(model_func(x_w_dummy_for_int), y).numpy()


def loss_from_script(beta, min_prec=0.9, lmbda=100):
    torch_beta = torch.from_numpy(beta).float()
    model_func = lambda x:  x @ torch_beta

    x_w_dummy_for_int = np.column_stack((x, np.ones([x.shape[0]])))
    x_w_dummy_for_int = torch.from_numpy(x_w_dummy_for_int).float()
    
    f = model_func(x_w_dummy_for_int)
    
    loss = MinPrecLoss(
        min_prec = min_prec,
        lmbda = lmbda,
        sigmoid_hyperparams = {"gamma": 7, "delta": 0.035},
        sigmoid_params = {'mtilde': 6.85,'btilde': -3.54, 'mhat': 6.85, 'bhat': 1.59}
    )
    
    return loss.forward(f, y).numpy()


def logloss(beta, min_prec=0.9, lmbda=100):
    torch_beta = torch.from_numpy(beta).float()
    model_func = lambda x:  x @ torch_beta

    x_w_dummy_for_int = np.column_stack((x, np.ones([x.shape[0]])))
    x_w_dummy_for_int = torch.from_numpy(x_w_dummy_for_int).float()
    
    f = model_func(x_w_dummy_for_int)
    
    loss = MinPrecLossLogForm(
        min_prec = min_prec,
        lmbda = lmbda,
        sigmoid_hyperparams = {"gamma": 7, "delta": 0.035},
        sigmoid_params = {'mtilde': 6.85,'btilde': -3.54, 'mhat': 6.85, 'bhat': 1.59}
    )
    
    return loss.forward(f, y).numpy()


def leaky_loss(beta, min_prec=0.9, lmbda=100, lmbda2=1000, leaky_slope=0.01):
    torch_beta = torch.from_numpy(beta).float()
    model_func = lambda x:  x @ torch_beta

    x_w_dummy_for_int = np.column_stack((x, np.ones([x.shape[0]])))
    x_w_dummy_for_int = torch.from_numpy(x_w_dummy_for_int).float()
    
    f = model_func(x_w_dummy_for_int)
    
    loss = MinPrecLeakyLoss(
        min_prec = min_prec,
        lmbda = lmbda,
        lmbda2 = lmbda2,
        leaky_slope=leaky_slope
    )
    
    return loss.forward(f, y).numpy()

# +
# def get_loss_from_new_form(beta, min_prec=0.9, lmbda= 1e-3):
#     min_prec = min_prec
    
#     torch_beta = torch.from_numpy(beta).float()
#     model_func = lambda x:  x @ torch_beta

#     x_w_dummy_for_int = np.column_stack((x, np.ones([x.shape[0]])))
#     x_w_dummy_for_int = torch.from_numpy(x_w_dummy_for_int).float()
    
#     f = model_func(x_w_dummy_for_int)
    
#     #Paragraph below eqn 14 in paper
#     gamma = torch.tensor(7.)
#     delta = torch.tensor(0.035)

#     mtilde = torch.tensor(6.85)
#     btilde = torch.tensor(-3.54)

#     mhat = torch.tensor(6.85)
#     bhat = torch.tensor(1.59)
    
    
#     f_tilde_log = torch.log((1 + gamma * delta)/(1 + torch.exp(mtilde * f + btilde)))
#     f_hat_log = torch.log((1 + gamma * delta)/(1 + torch.exp(mhat * f + bhat)))

#     return -torch.sum(-(lmbda + 1 )*y*f_tilde_log + (lmbda*min_prec/(1-min_prec))*(1-y)*f_hat_log +  lmbda*gamma * delta * y).numpy()
# -

# # Plotting

def get_loss_landscape(loss_function, num_samples, w0_width, w1_width, kwargs={}):
    N = num_samples
    xv, yv = np.meshgrid(
        sklearnbetas[0] + np.linspace(-w0_width, w0_width, N), 
        sklearnbetas[1] + np.linspace(-w1_width, w1_width, N)
    )
    input_params = np.column_stack([xv.ravel(), yv.ravel(), sklearnbetas[2]*np.ones(N*N)])
    # input_params = np.column_stack([xv.ravel(), yv.ravel(), sklearnbetas[2]*np.random.normal(1, 0.1, N*N)])

    losses = np.apply_along_axis(loss_function, 1, input_params, **kwargs)
    
    data = [
        go.Surface(z=losses.reshape(xv.shape), x=xv, y=yv),
        go.Scatter3d(x = [sklearnbetas[0]], y = [sklearnbetas[1]], z = [loss_function(sklearnbetas, **kwargs)], mode='markers',marker=dict(size=6, color='black'))
    ]

    layout = go.Layout(
        # autosize=False,
        width=800,
        height=800,
    )
    return go.Figure(data=data, layout=layout)


iplot(get_loss_landscape(loss_from_script, 80, 3, 2, {'min_prec': 0.9, 'lmbda': 1e4}))

iplot(
    get_loss_landscape(
        leaky_loss, 80, 3, 2, 
        {'min_prec': 0.9, 'lmbda': 1e4, 'lmbda2': 1e6, 'leaky_slope':0.001}
    )
)

# # BCE tests

get_loss_landscape(loss_for_bce, 80, 2, 2, )

get_loss_landscape(loss_for_bce_manual, 250, 5, 5, )





# # losses

# +
x = np.linspace(-10, 10, 1000)

zero_one = 1 - np.heaviside(x, 0)

sigmoid = lambda xx: 1/(1+np.exp(-xx))

sig_loss = 1 - sigmoid(x)

logloss = - np.log(sigmoid(x))
logloss_base_n = - np.log(sigmoid(x-10)) / np.log(10000)

# piecewise_relu = np.minimum(-0.001*x+1, np.maximum(0, 1-x))
neg_leaky_relu = np.where(x > 0, 1-x, 1 - x * 0.01)
relu_of_leaky_relu = np.maximum(neg_leaky_relu, 0)



plt.plot(x, zero_one, label='0-1')
# plt.plot(x, sig_loss, label='sig_loss')
# plt.plot(x, logloss, label='logloss')
# plt.plot(x, logloss_base_n, label='logloss_base_n')
plt.plot(x, relu_of_leaky_relu, label='relu_of_leaky_relu')
plt.ylim(-1, 3)
plt.legend()

# +
plt.plot(x, zero_one, label='0-1')

torch_x = torch.linspace(-10, 10, 100)
torch_relu = nn.ReLU()( torch_x)
torch_relu_leaky_relu = nn.ReLU()(1 - nn.LeakyReLU(negative_slope=0.01)(torch_x))
torch_leaky_relu_leaky_relu = nn.LeakyReLU(negative_slope=-0.01)(1 - nn.LeakyReLU(negative_slope=0.01)(torch_x))
loglogsigmoid = -torch.log(torch.sigmoid(torch_x))

plt.plot(torch_x.numpy(), loglogsigmoid.numpy())
# -


