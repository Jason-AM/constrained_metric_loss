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
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# +
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import fbeta_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from constrained_metric_loss.min_precision_loss import (
    MinPrecLoss, LearnableMinPrecLoss, MinPrecLeakyLoss
)

from constrained_metric_loss.linear_model import LinearModel, InitialParameterGenerator

import openml

# from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objects as go

# init_notebook_mode()
# -

np.set_printoptions(suppress=True)

# # Dataset Preperation

all_data_dict = {}

# ### generated data

# +
n = 10000

rng = np.random.default_rng(5)
x1 = rng.normal(0, 1, size=n)
x2 = rng.normal(3, 2.2, size=n)

beta0_true = -0.4
beta1_true = -5.3
beta2_true = 3.1

p = 1 / (1 + np.exp(-(beta0_true + beta1_true * x1 + beta2_true * x2)))

y = rng.binomial(1, p, size=n)
x = np.column_stack((x1, x2))

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42, stratify=y)

all_data_dict["basic"] = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


# +
from sklearn.datasets import make_moons

x, y = make_moons(n_samples=10000, noise=0.1, random_state=7)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42, stratify=y)

all_data_dict["moons"] = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}

# +
from data import toydata

x, y, _, _, _, _ = toydata.create_toy_dataset()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42, stratify=y)

all_data_dict["paper_data"] = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


# -

# ### real data from openml


def prepare_openml_data(dataset_name, scale_data):
    data = openml.datasets.get_dataset(dataset_name)
    X_all, _, _, _ = data.get_data(dataset_format="dataframe")
    X_all = X_all.dropna(how="any")

    X = X_all.iloc[:, :-1].values
    y = X_all.iloc[:, -1].values.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)

    if scale_data:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


# +

all_data_dict = all_data_dict | {
    "creditcard": prepare_openml_data("creditcard", scale_data=True),
    "higgs": prepare_openml_data("higgs", scale_data=False),
    "global_equity": prepare_openml_data("numerai28.6", scale_data=False),
    "diabities": prepare_openml_data("Diabetes-Data-Set", scale_data=True),
}


# -

# # run tests


# +
def training_w_lambda_grad_ascent(model, data_dict, n_batches, num_lambda_grad_asscent):
    list_of_model_params = list(model.parameters())
    
    model_params = list_of_model_params[:-num_lambda_grad_asscent]
    optimizer = optim.Adam(model_params, lr=0.1)

    lam = list_of_model_params[-num_lambda_grad_asscent:]
    optimizer_lam = optim.Adam(lam, lr=0.01)

    training_losses = model.fit_w_lambda_grad_ascent(
        data_dict["X_train"], 
        data_dict["y_train"],
        optimizer, 
        optimizer_lam,
        n_batches=n_batches
    )
    
    return model, training_losses

def training_all_grad_descent(model, data_dict, n_batches):
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    training_losses = model.fit(
        data_dict["X_train"], 
        data_dict["y_train"].reshape(-1,1),
        optimizer, 
        n_batches=n_batches
    )
    
    return model, training_losses


def run_model(
    data_dict,
    model,
    n_batches=200,
    num_lambda_grad_asscent=0,
    thresh = 0.5
):
    
    if num_lambda_grad_asscent==0:
        model, training_losses = training_all_grad_descent(
            model, data_dict, n_batches
        )
    else:
        model, training_losses = training_w_lambda_grad_ascent(
            model, data_dict, n_batches, num_lambda_grad_asscent
        )
    
    

    phat = model.sigmoid_of_score(data_dict["X_test"])
    yhat = (phat >= thresh).astype(int)

    prec = precision_score(data_dict['y_test'], yhat, zero_division=0)
    rec = recall_score(data_dict['y_test'], yhat, zero_division=0)

    return prec, rec, training_losses, model
# -

# ### performance comparisons

all_data_dict.keys()

data_dict = all_data_dict["paper_data"]
target_prec = 0.8


def optimal_prec_recall_search(target_prec, prec_recall_list, tol=0.02, beta=1):
    prec_recall_arr = np.array(prec_recall_list)
    diff_to_target = prec_recall_arr[:,0] - target_prec
    candidates = prec_recall_arr[diff_to_target>-tol]
    fbeta_score = (1+beta**2) * (candidates[:,0] * candidates[:,1])/((beta**2)*candidates[:,0] + candidates[:,1])
    return candidates[np.argsort(fbeta_score)[::-1]]


# +
# %%time

sklearnlogreg = LogisticRegression()
sklearnlogreg = sklearnlogreg.fit(data_dict["X_train"], data_dict["y_train"])

phat = sklearnlogreg.predict_proba(data_dict["X_test"])[:, 1]

prec_recalls = []
for thresh in np.arange(0., 1.01, 0.01):
    
    yhat = (phat >= thresh).astype(int)

    prec = precision_score(data_dict["y_test"], yhat, zero_division=0)
    rec = recall_score(data_dict["y_test"], yhat, zero_division=0)

    prec_recalls.append([prec, rec])
    
optimal_prec_recall_search(target_prec, prec_recalls)[:15]

# +
param_seed = np.concatenate([sklearnlogreg.coef_.flatten(), sklearnlogreg.intercept_])

initial_param_gen = InitialParameterGenerator(param_seed)
initial_param_gen.generate_samples_from_grid(num_to_select=5, number_per_dim=10, seed_multiplier=10)
initial_param_gen.generate_normally_distributed_around_seed(number=5, var=5)
initial_param_gen.generate_normally_distributed(number=5, mean=0, var=5)

param_inits = initial_param_gen.initial_parameters
# -

# ### min prec loss w linear model

# + tags=[]
# %%time

num_lambda_grad_asscent = 1
train_lambda_with_ascent = num_lambda_grad_asscent != 0

num_non_conv = 0
prec_recalls = []

for param_i, param_init in enumerate(param_inits):
    
    model = LinearModel(
        nfeat=data_dict["X_train"].shape[1],
        model_param_init=param_init,
        loss=MinPrecLoss,
        loss_arguments={
            "min_prec": target_prec,
            "lmbda":  nn.Parameter(torch.tensor([1e1]), requires_grad=train_lambda_with_ascent),
            "sigmoid_hyperparams": {"gamma": 7, "delta": 0.035, "eps": 0.75},
        },
    )

    maxrecall_prec, maxrecall_recall, training_losses, _ = run_model(
        data_dict,
        model,
        n_batches= 2_000,
        num_lambda_grad_asscent = num_lambda_grad_asscent
    )
    
    prec_recalls.append([maxrecall_prec, maxrecall_recall, int(param_i)])
    num_non_conv += int(np.array(training_losses)[-1] > 0.05*np.array(training_losses)[0])
    plt.plot(training_losses)
plt.show()
print(num_non_conv/ len(param_inits))  
optimal_prec_recall_search(target_prec, prec_recalls, tol=0.02)[:15]
# -


# ### test new loss forms w linear model

# +

# %%time
num_lambda_grad_asscent = 2
train_lambda_with_ascent = num_lambda_grad_asscent != 0


num_non_conv = 0
prec_recalls = []
leaky_grads = [0.01] #np.logspace(-1, -3, 20)
for param_i, param_init in enumerate(param_inits):
    for ls in leaky_grads:
        
        model = LinearModel(
            nfeat=data_dict["X_train"].shape[1],
            model_param_init=param_init,
            loss=MinPrecLeakyLoss,
            loss_arguments={
                "min_prec": target_prec, 
                "lmbda": nn.Parameter(torch.tensor([1e1]), requires_grad=train_lambda_with_ascent), 
                'lmbda2': nn.Parameter(torch.tensor([1e3]), requires_grad=train_lambda_with_ascent), 
                'leaky_slope': ls
            },
        )
        
        new_prec, new_recall, new_training_losses, model = run_model(
            data_dict,
            model,
            n_batches= 2_000,
            num_lambda_grad_asscent = num_lambda_grad_asscent
        )

        lambdas = [ i.detach() for i in model.parameters()][1:]
        prec_recalls.append([new_prec, new_recall, int(param_i), ls, lambdas])
        num_non_conv += int(np.array(new_training_losses)[-1] > 0.05*np.array(new_training_losses)[0])
        plt.plot(new_training_losses)
plt.show()
print(num_non_conv/ (len(param_inits)*len(leaky_grads)))  
    
optimal_prec_recall_search(target_prec, prec_recalls, tol=0.02)[:15]


# -
# ### MLP 

# +
from constrained_metric_loss.mlp import MLP



num_lambda_grad_asscent = 0
train_lambda_with_ascent = num_lambda_grad_asscent != 0

num_initialisations = 20

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.uniform_(m.bias,-1, 1)


prec_recalls = []

for _ in range(num_initialisations):
        
    # model = MLP(
    #     nfeat=data_dict["X_train"].shape[1], 
    #     hidden_width=4, 
    #     loss=MinPrecLoss,
    #     loss_arguments={
    #         "min_prec": target_prec,
    #         "lmbda":  nn.Parameter(torch.tensor([1e2]), requires_grad=train_lambda_with_ascent),
    #         "sigmoid_hyperparams": {"gamma": 7, "delta": 0.035, "eps": 0.75},
    #     },
    # )
    
    # model = MLP(
    #     nfeat=data_dict["X_train"].shape[1],
    #     hidden_width=4, 
    #     loss=MinPrecLeakyLoss,
    #     loss_arguments={
    #         "min_prec": target_prec, 
    #         "lmbda": nn.Parameter(torch.tensor([1e2]), requires_grad=train_lambda_with_ascent), 
    #         'lmbda2': nn.Parameter(torch.tensor([1e1]), requires_grad=train_lambda_with_ascent), 
    #         'leaky_slope': 0.1
    #     },
    # )
    
    model = MLP(
        nfeat=data_dict["X_train"].shape[1],
        hidden_width=4, 
        loss=nn.HingeEmbeddingLoss,
        loss_arguments={},
    )
    
    model.apply(init_weights)

    new_prec, new_recall, new_training_losses, model = run_model(
        data_dict,
        model,
        n_batches= 1_000,
        num_lambda_grad_asscent = num_lambda_grad_asscent
    )
    
    if num_lambda_grad_asscent !=0:
        lambdas = [ i.detach() for i in model.parameters()][-num_lambda_grad_asscent:]
        prec_recalls.append([new_prec, new_recall, lambdas])
    else:
        prec_recalls.append([new_prec, new_recall])
        
    num_non_conv += int(np.array(new_training_losses)[-1] > 0.05*np.array(new_training_losses)[0])
    plt.plot(new_training_losses)
plt.show()

optimal_prec_recall_search(target_prec, prec_recalls, tol=0.02)[:15]
# -


