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


class classification_model_score(nn.Module):
    """Produces classification scores for given loss"""

    def __init__(
        self,
        nfeat,
        model_param_init,
        loss,
        loss_arguments,
    ):
        super().__init__()

        self.nfeat = nfeat
        self.beta = nn.Parameter(torch.from_numpy(model_param_init).float())
        assert nfeat + 1 == len(model_param_init), "Using the coefficient trick so need to add one extra parameter"

        self.loss = loss
        self.loss_arguments = loss_arguments

        self.loss_func = self.loss(**self.loss_arguments)

    def forward(self, x, y):
        x = np.column_stack((x, np.ones([x.shape[0]])))
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y)

        self.f = x @ self.beta

        return self.loss_func(self.f, y.float())

    def sigmoid_of_score(self, xtest):

        if len(xtest.shape) == 1:
            xtest = xtest.reshape([1, self.nfeat])

        xtest = np.column_stack((xtest, np.ones([xtest.shape[0]])))
        xtest = torch.from_numpy(xtest).float()

        f = xtest @ self.beta
        return torch.sigmoid(f).detach().numpy().flatten()

    def fit(self, x, y, optimizer, n_batches=200):
        training_loss = []
        for i in range(n_batches):
            optimizer.zero_grad()
            loss = self.forward(x, y)
            loss.backward()
            optimizer.step()
            
            training_loss.append(loss.detach().numpy())

            
        return training_loss


def run_model(
    data_dict,
    thresh,
    model_param_init,
    loss,
    loss_params,
    n_batches=200,
):

    model = classification_model_score(
        nfeat=data_dict["X_train"].shape[1],
        model_param_init=model_param_init,
        loss=loss,
        loss_arguments=loss_params,
    )

    optimizer = optim.Adam(model.parameters(), lr=0.1)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.1)

    training_losses = model.fit(data_dict["X_train"], data_dict["y_train"], optimizer, n_batches=n_batches)

    phat = model.sigmoid_of_score(data_dict["X_test"])
    yhat = (phat >= thresh).astype(int)

    prec = precision_score(data_dict['y_test'], yhat, zero_division=0)
    rec = recall_score(data_dict['y_test'], yhat, zero_division=0)

    return prec, rec, training_losses, model


# ### performance comparisons

all_data_dict.keys()

data_dict = all_data_dict["creditcard"]
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
def full_grid(seed_vals, number_per_dim, seed_multiplier = 10):

    linspaces = [
        np.linspace(-seed_multiplier*seed, seed_multiplier*seed, number_per_dim) 
        for seed in seed_vals
    ]
    n_times_away_grid = np.meshgrid(*linspaces)
    search_space = [dim_params.flatten() for dim_params in n_times_away_grid]
    
    return np.column_stack(search_space)

def random_sampling_from_grid(
    seed_vals, number_per_dim, seed_multiplier = 10, num_to_select=50
):
    rng = np.random.default_rng(0)
    return rng.choice(
        full_grid(
            seed_vals, 
            number_per_dim=number_per_dim, 
            seed_multiplier=seed_multiplier
        ), 
        size = num_to_select
    )
    

def linear_grid(seed_vals, number_per_dim, seed_multiplier = 10):
    
    linspaces = [
        np.linspace(-seed_multiplier*seed, seed_multiplier*seed, number_per_dim) 
        for seed in seed_vals
    ]
    
    return np.column_stack(linspaces)




# +
param_seed = np.concatenate([sklearnlogreg.coef_.flatten(), sklearnlogreg.intercept_])

# param_inits = random_sampling_from_grid(
#     param_seed, number_per_dim=2, seed_multiplier=10, num_to_select=100
# ) #full_grid(param_seed, 4, 10) 

param_inits = linear_grid(param_seed, 50, 100)

param_inits = np.vstack([param_inits, param_seed])

for seed in [1,2,3,4,5,6]:
    rng = np.random.default_rng(seed)
    param_inits = np.vstack([param_inits, param_seed + rng.normal(0, 1, len(param_seed))])
    param_inits = np.vstack([param_inits, rng.normal(0, 1, len(param_seed))])

# + tags=[]
# %%time
num_non_conv = 0
prec_recalls = []
for param_i, param_init in enumerate(param_inits):

    maxrecall_prec, maxrecall_recall, training_losses, _ = run_model(
        data_dict,
        0.5,
        param_init,
        loss=MinPrecLoss,
        loss_params={
            "min_prec": target_prec,
            "lmbda":  1e4,
            "sigmoid_hyperparams": {"gamma": 7, "delta": 0.035, "eps": 0.75},
            # 'sigmoid_params': {'mtilde': 6.85,'btilde': -3.54, 'mhat': 6.85, 'bhat': 1.59}
        },
        n_batches= 500
    )
    
    prec_recalls.append([maxrecall_prec, maxrecall_recall, int(param_i)])
    num_non_conv += int(np.array(training_losses)[-1] > 0.05*np.array(training_losses)[0])
    plt.plot(training_losses)
plt.show()
print(num_non_conv/ len(param_inits))  
optimal_prec_recall_search(target_prec, prec_recalls, tol=0.02)[:15]


# +

# %%time
num_non_conv = 0
prec_recalls = []
leaky_grads = [0.05, 0.01] #np.logspace(-1, -3, 20)
for param_i, param_init in enumerate(param_inits):
    for ls in leaky_grads:
        new_prec, new_recall, new_training_losses, model = run_model(
            data_dict,
            0.5,
            param_init,
            loss=MinPrecLeakyLoss,
            # loss_params={"min_prec": target_prec, "lmbda": 1e7, 'lmbda2': 1e8, 'leaky_slope': 0.01},
            loss_params={
                "min_prec": target_prec, 
                "lmbda": nn.Parameter(torch.tensor([1e4]), requires_grad=True), 
                'lmbda2': nn.Parameter(torch.tensor([1e3]), requires_grad=True), 
                'leaky_slope': ls
            },
            n_batches= 500
        )

        lambdas = [ i.detach() for i in model.parameters()][1:]
        prec_recalls.append([new_prec, new_recall, int(param_i), ls, lambdas])
        num_non_conv += int(np.array(new_training_losses)[-1] > 0.05*np.array(new_training_losses)[0])
        plt.plot(new_training_losses)
plt.show()
print(num_non_conv/ (len(param_inits)*len(leaky_grads)))  
    
optimal_prec_recall_search(target_prec, prec_recalls, tol=0.02)[:15]


# -





