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

from constrained_metric_loss.min_precision_loss import MinPrecLoss

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
# -

# #### params from sklearn

# +

sklearnlogreg = LogisticRegression()
sklearnlogreg = sklearnlogreg.fit(x,y)
sklearnbetas = np.concatenate([sklearnlogreg.coef_.flatten(), sklearnlogreg.intercept_])
# -

sklearnbetas


# # Losses

def get_loss_for_bce(beta):
    torch_beta = torch.from_numpy(beta).float()
    model_func = lambda xx:  xx @ torch_beta

    x_w_dummy_for_int = np.column_stack((x, np.ones([x.shape[0]])))
    x_w_dummy_for_int = torch.from_numpy(x_w_dummy_for_int).float()

    loss_funct = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.01))

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
        go.Scatter3d(x = [sklearnbetas[0]], y = [sklearnbetas[1]], z = [loss_function(sklearnbetas, **kwargs)], mode='markers',marker=dict(size=12, color='black'))
    ]

    layout = go.Layout(
        # autosize=False,
        width=800,
        height=800,
    )
    return go.Figure(data=data, layout=layout)


iplot(get_loss_landscape(loss_from_script, 40, 4, 4, {'min_prec': 0.9, 'lmbda': 1e-3}))


