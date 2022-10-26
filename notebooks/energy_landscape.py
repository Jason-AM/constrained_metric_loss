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
    MinPrecLoss, MinPrec01LossApprox, ZeroOneApproximation
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

def get_loss_landscape(
    loss_function, num_samples, x_lims, y_lims, kwargs={}
):
    N = num_samples
    xv, yv = np.meshgrid(
        sklearnbetas[0] + np.linspace(x_lims[0], x_lims[1], N), 
        sklearnbetas[1] + np.linspace(y_lims[0], y_lims[1], N)
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


# +
fig = get_loss_landscape(
    loss_from_script, 
    num_samples = 120, 
    x_lims = (-1, 3.5), #(1,2.1), 
    y_lims = (-2, 2), #(-1,1), 
    kwargs = {'min_prec': 0.8, 'lmbda': 1e2},
)
fig.update_layout(scene = dict(zaxis = dict(range=[2_000, 20_000])))

iplot(fig)
# -

fig = get_loss_landscape(
    leaky_loss, 
    120,
    x_lims = (-1.5, 3.5), #(1,2.1), 
    y_lims = (-2, 1.5), #(-1,1), 
    kwargs = {
        'min_prec': 0.8, 'lmbda': 1e2, 'lmbda2': 2e2, 'leaky_slope':0.001
    }
)
fig.update_layout(scene = dict(zaxis = dict(range=[15_000, 35_000])))
iplot(fig)





# # overlay losses

def get_loss_landscape_overlayed(
    loss_function1, 
    loss_function1_args, 
    loss_function2, 
    loss_function2_args, 
    num_samples, 
    x_lims, 
    y_lims,
    plot_surface_diff = False
):
    N = num_samples
    xv, yv = np.meshgrid(
        sklearnbetas[0] + np.linspace(x_lims[0], x_lims[1], N), 
        sklearnbetas[1] + np.linspace(y_lims[0], y_lims[1], N)
    )
    input_params = np.column_stack(
        [xv.ravel(), yv.ravel(), sklearnbetas[2]*np.ones(N*N)]
    )
    
    losses1 = np.apply_along_axis(
        loss_function1, 1, input_params, **loss_function1_args
    )
    
    
    losses2 = np.apply_along_axis(
        loss_function2, 1, input_params, **loss_function2_args
    )
    
    
    if plot_surface_diff:
        loss_diff = losses1 - losses2
        surface_diff = go.Surface(z=loss_diff.reshape(xv.shape), x=xv, y=yv)
        data = [surface_diff]
    else:
        surface1 = go.Surface(z=losses1.reshape(xv.shape), x=xv, y=yv)
        surface2 = go.Surface(z=losses2.reshape(xv.shape), x=xv, y=yv)
        data = [
            surface1,
            surface2,
        ]

    layout = go.Layout(
        # autosize=False,
        width=800,
        height=800,
    )
    return go.Figure(data=data, layout=layout)

# +
fig = get_loss_landscape_overlayed(
    loss_function1 = loss_from_script, 
    loss_function1_args =  {'min_prec': 0.8, 'lmbda': 1e4}, 
    loss_function2 = leaky_loss, 
    loss_function2_args = {'min_prec': 0.8, 'lmbda': 1e4, 'lmbda2': 0.0, 'leaky_slope':0.001}, 
    num_samples = 80,
    x_lims = (-1, 2.5), #(1,2.1), , 
    y_lims = (-1.5, 1.5), #(-1,1), ,
    plot_surface_diff = False
)

iplot(fig)
# -



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





# +



def zero_one_loss(xx, y):
    return 1 - torch.heaviside((2 * y - 1) * xx, torch.tensor([0.5])).numpy()

semi_rath_hughes_loss = ZeroOneApproximation(
    name_of_approx = "semi_rath_hughes_approx",
    params_dict = {'gamma': 7.0, 'delta': 0.035, 'eps': 0.75}
)

logsigmoid_loss = ZeroOneApproximation(
    name_of_approx = "logsigmoid_approx",
    params_dict = {}
)

leaky_relu_loss = ZeroOneApproximation(
    name_of_approx = "leaky_relu_approx",
    params_dict = {"negative_slope": 0.01}
)

hinge_loss = ZeroOneApproximation(
    name_of_approx = "leaky_relu_approx",
    params_dict = {"negative_slope": 1.}
)


xx = torch.linspace(-2.5, 2.5, 1000)

plt.plot(
    xx, zero_one_loss(xx, 1), label='zero_one_loss'
)
plt.plot(
    xx, semi_rath_hughes_loss.calc_loss(xx, 1), label='semi_rath_hughes'
)
plt.plot(
    xx, logsigmoid_loss.calc_loss(xx, 1), label='logsigmoid_loss'
)
plt.plot(
    xx, leaky_relu_loss.calc_loss(xx, 1), label='leaky_relu_loss'
)
plt.plot(
    xx, hinge_loss.calc_loss(xx, 1), label='hinge_loss'
)
plt.legend()
# -


