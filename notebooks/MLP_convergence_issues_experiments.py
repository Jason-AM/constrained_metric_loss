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
#     display_name: Python 3.9.12 64-bit ('pydef')
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# +
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, roc_curve
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from constrained_metric_loss.min_precision_loss import MinPrecLoss

# +
#Choose MLP with BCE loss or with MinPrec loss

modelchosen = 'mlp' #'mlp' #'linear'
losschosen = 'minprec' #'bce' #'minprec'

# +
gamma = 7 #7
delta = 0.035 #0.035
eps = 0.75 #0.75

min_prec = 0.9

lmbda = 1e4

L2_coeff = 0.01
lr = 0.1

loss_params = {
        'min_prec':min_prec, 
        'lmbda': lmbda, 
        'sigmoid_hyperparams': {'gamma': gamma, 'delta': delta, 'eps': eps}
    }

1+gamma*delta


# -

# # model

class LinearScore(nn.Module):
    """Normal logistic regression with BCE loss"""
    def __init__(self, nfeat, model_param_init, loss, loss_arguments, trainable_loss_arguments=False):
        super().__init__()
        
        self.nfeat = nfeat
        self.beta = nn.Parameter(torch.from_numpy(model_param_init).float())
        assert nfeat + 1 == len(model_param_init), "Using the bias trick so need to add one extra parameter"
        
        self.loss = loss
        self.loss_arguments = loss_arguments
        self.trainable_loss_arguments = trainable_loss_arguments
        
        if not self.trainable_loss_arguments:
            self.loss_func = self.loss(**self.loss_arguments)

            
        
    def forward(self, x, y):
        x = np.column_stack((x, np.ones([x.shape[0]])))
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y)
        
        
        self.f = x @ self.beta
        
        if self.trainable_loss_arguments:
            self.loss_func = self.loss(**self.loss_arguments)
        
        return self.loss_func(self.f, y.float())
    
    def predict_proba(self, xtest):
        
        if len(xtest.shape) == 1:
            xtest = xtest.reshape([1, self.nfeat])
        
        xtest = np.column_stack((xtest, np.ones([xtest.shape[0]])))
        xtest = torch.from_numpy(xtest).float()
        
        f = xtest @ self.beta
        return torch.sigmoid(f).detach().numpy().flatten()
    
    def fit(self, x, y, optimizer, n_epochs=10):
        
        for i in range(n_epochs):
            optimizer.zero_grad()
            loss = self.forward(x, y)
            loss.backward()
            optimizer.step()


class MLP(nn.Module):
    """Multi-Layer Perceptron"""
    def __init__(self, hidden_width, loss, loss_arguments, trainable_loss_arguments=False):
        super().__init__()
        
        #self.nfeat = nfeat
        #self.beta = nn.Parameter(torch.from_numpy(model_param_init).float())
        #assert nfeat + 1 == len(model_param_init), "Using the bias trick so need to add one extra parameter"
        
        self.hidden_width = hidden_width
        self.loss = loss
        self.loss_arguments = loss_arguments
        self.trainable_loss_arguments = trainable_loss_arguments
        
        if not self.trainable_loss_arguments:
            self.loss_func = self.loss(**self.loss_arguments)
    
        self.layers = nn.Sequential(
          nn.Flatten(),
          nn.Linear(2, self.hidden_width),
          #nn.BatchNorm1d(num_features=self.hidden_width),
          nn.Sigmoid(),
          #nn.ReLU(),
          nn.Linear(self.hidden_width, 1)
        )
            
        
    def forward(self, x, y):
        #x = np.column_stack((x, np.ones([x.shape[0]])))
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y)        
        
        #self.f = x @ self.beta
        self.f = self.layers(x)
        
        if self.trainable_loss_arguments:
            self.loss_func = self.loss(**self.loss_arguments)
        
        return self.loss_func(self.f, y.float())
    
    def predict_proba(self, xtest):
        #self.eval()
        
        #if len(xtest.shape) == 1:
        #    xtest = xtest.reshape([1, self.nfeat])
        
        #xtest = np.column_stack((xtest, np.ones([xtest.shape[0]])))
        xtest = torch.from_numpy(xtest).float()
        
        #f = xtest @ self.beta
        f = self.layers(xtest)
        return torch.sigmoid(f).detach().numpy().flatten()
    
    def fit(self, x, y, optimizer, n_epochs=10):
        
        for i in range(n_epochs):
            optimizer.zero_grad()
            loss = self.forward(x, y)
            loss.backward()
            optimizer.step()


#https://discuss.pytorch.org/t/how-to-initialize-weights-in-nn-sequential-container/8534/2?u=ilanfri
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 1.0)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def decisionLine(model, testdata, thresh=0.5, eps=0.05, gridsize=1000):
    
    """Takes a trained model with a predict_proba() method, and returns the coordinates
    for the decision boundary, where the boundary is set by points where the predicted
    probability is within eps of the threshold thresh.
    
    This is computed on a grid of size gridsize x gridsize, which covers the range 
    of testdata
    
    Function is intended only to plot decision boundary for 2D datasets
    
    Inputs:
    model: trained model with predict_proba() method
    testdata: the test data (only used to compute grid boundaries)
    thresh: probability threshold at which decision boundary is set
    eps: tolerance around the threshold (exact value thresh is rarely predicted)
    gridsize: number of points to predict on for grid, per dimension
    
    Output
    2D coordinates of decision boundary line
    """
    
    xmin = testdata[:, 0].min()
    xmax = testdata[:, 0].max()

    ymin = testdata[:, 1].min()
    ymax = testdata[:, 1].max()

    xx, yy = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))

    grid = np.column_stack([np.ravel(xx), np.ravel(yy)])
    
    p_grid = model.predict_proba(grid)
    
    decision_line = grid[np.where(np.abs(p_grid-thresh) <= eps)]
    
    return decision_line


# ### Logistic regression dataset

# +
n = 10000

rng = np.random.default_rng(5)

x1 = rng.normal(0, 1, size=n)
x2 = rng.normal(3, 2.2, size=n)


beta0_true = -0.4
beta1_true = -5.3
beta2_true = 3.1

p = 1/(1+np.exp(-(beta0_true + beta1_true*x1 + beta2_true*x2)))

y_basic = rng.binomial(1, p, size=n)
x_basic = np.column_stack((x1, x2))

# +
#plt.scatter(x1, x2, c=p)

# +
ntest = int(np.floor(n/10))

x1test = rng.normal(0, 1, size=ntest)
x2test = rng.normal(3, 2.2, size=ntest)

ptest = 1/(1+np.exp(-(beta0_true + beta1_true*x1test + beta2_true*x2test)))

xtest_basic = np.column_stack((x1test, x2test))

ytest_basic = rng.binomial(1, ptest, size=ntest)

# + slideshow={"slide_type": "slide"}
import torch
import torch.nn as nn
import torch.optim as optim

class tight_sigmoid_upper(nn.Module):
    def __init__(self, eps, delta, gamma):
        super().__init__()
        
        self.eps = eps
        self.delta = delta
        self.gamma = gamma
        
        self.m = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))
        
    def forward(self):
        
        def u(a):
            return (1+self.gamma*self.delta)*torch.sigmoid(self.m*a + self.b)
        
        obj = torch.sum(torch.square(self.delta - u(-self.eps)) + torch.square(1 + self.delta - u(0)))
        return obj


# + slideshow={"slide_type": "skip"}
model_upper = tight_sigmoid_upper(eps, delta, gamma)

optimizer_upper = optim.Adam(model_upper.parameters(), lr=1)

for _ in range(1000):
    optimizer_upper.zero_grad()
    loss = model_upper()
    loss.backward()
    optimizer_upper.step()

# + slideshow={"slide_type": "-"}
m_upper, b_upper = model_upper.named_parameters()
m_upper = float(m_upper[1].detach().numpy())
b_upper = float(b_upper[1].detach().numpy())

print((m_upper, b_upper))
# + slideshow={"slide_type": "slide"}
import torch
import torch.nn as nn
import torch.optim as optim

class tight_sigmoid_lower(nn.Module):
    def __init__(self, eps, delta, gamma):
        super().__init__()
        
        self.eps = eps
        self.delta = delta
        self.gamma = gamma
        
        self.m = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))
        
    def forward(self):
        
        def u(a):
            return (1+self.gamma*self.delta)*torch.sigmoid(self.m*a + self.b)
        
        obj = torch.sum(torch.square(self.delta - u(0)) + torch.square(1 + self.delta - u(self.eps)))
        return obj


# + slideshow={"slide_type": "skip"}

model_lower = tight_sigmoid_lower(eps, delta, gamma)

optimizer_lower = optim.Adam(model_lower.parameters(), lr=1)

for _ in range(1000):
    optimizer_lower.zero_grad()
    loss = model_lower()
    loss.backward()
    optimizer_lower.step()

# + slideshow={"slide_type": "-"}
m_lower, b_lower = model_lower.named_parameters()
m_lower = float(m_lower[1].detach().numpy())
b_lower = float(b_lower[1].detach().numpy())

print((m_lower, b_lower))
# +
from scipy.special import expit

def bound_sigmoid(x, m=1, b=0, eps=0.75, delta=0.035, gamma=7.):
    return (1+gamma*delta)*expit(m*x + b)


# + slideshow={"slide_type": "-"}
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 1, 100)

y_tpc = np.where(x<0, 0, 1) + gamma*delta
y_tpc_bound = bound_sigmoid(x, m=m_lower, b=b_lower)

y_fpc = np.where(x<0, 0, 1)
y_fpc_bound = bound_sigmoid(x, m=m_upper, b=b_upper)

plt.plot(x, y_tpc, drawstyle='steps', linestyle='dotted', color='blue')
plt.plot(x, y_tpc_bound, color='blue', label='tpc')

plt.plot(x, y_fpc, drawstyle='steps', linestyle='dotted', color='green')
plt.plot(x, y_fpc_bound, color='green', label='fpc')

plt.xlabel(r'$f_\theta(x)-b$')
plt.ylabel(r'$\hat{y}$')

plt.legend()
plt.show()

# +
if modelchosen=='mlp' and losschosen == 'minprec':
    mlp_minprec_model = MLP(hidden_width=2, loss=MinPrecLoss, loss_arguments=loss_params)
elif modelchosen == 'mlp' and losschosen == 'bce':
    mlp_minprec_model = MLP(hidden_width=2, loss=nn.BCEWithLogitsLoss, loss_arguments={})
    y_basic = y_basic[:, np.newaxis]
elif modelchosen == 'linear' and losschosen == 'bce':
    param_init = np.zeros(x_basic.shape[1] + 1)
    mlp_minprec_model = LinearScore(nfeat=x_basic.shape[1], model_param_init=param_init, loss=nn.BCEWithLogitsLoss, loss_arguments={})
elif modelchosen == 'linear' and losschosen == 'minprec':
    param_init = np.zeros(x_basic.shape[1] + 1)
    mlp_minprec_model = LinearScore(nfeat=x_basic.shape[1], model_param_init=param_init, loss=MinPrecLoss, loss_arguments=loss_params)


#mlp_minprec_model.apply(weights_init)


optimizer_minprec = optim.Adam(mlp_minprec_model.parameters(), lr=lr, weight_decay=L2_coeff)
mlp_minprec_model.fit(x_basic, y_basic, optimizer_minprec)
    
phat_minprec = mlp_minprec_model.predict_proba(xtest_basic)
#yhat = (phat >= 0.5).astype(int)
    
#prec = precision_score(ytest_toy, yhat)
#rec = recall_score(ytest_toy, yhat)

#prec, rec

np.random.choice(phat_minprec,10)
# -

decision_line = decisionLine(mlp_minprec_model, xtest_basic)
decision_line[:10]

# +
cdict = {0: 'green', 1: 'blue'}
mdict = {0:'o', 1:'x'}

fig, ax = plt.subplots()
for g in np.unique(ytest_basic):
    ix = np.where(ytest_basic == g)
    ax.scatter(xtest_basic[ix,0], xtest_basic[ix,1], c = cdict[g], marker=mdict[g], label = g, alpha=0.3)
ax.legend()

ax.plot(decision_line[:,0], decision_line[:, 1], c='red')

plt.show()

# +
yhat = (phat_minprec >= 0.5).astype(int)
    
prec = precision_score(ytest_basic, yhat)
rec = recall_score(ytest_basic, yhat)

prec, rec
# -
# ### moons dataset

# +
from sklearn.datasets import make_moons

xmoons, ymoons = make_moons(n_samples = 10000, noise=0.1, random_state=7)

# +
xmoons_train = xmoons[:9000]
ymoons_train = ymoons[:9000]

xmoons_test = xmoons[9000:]
ymoons_test = ymoons[9000:]

# +
#plt.scatter(xmoons_train[:,0], xmoons_train[:,1], c=ymoons_train, alpha=0.3)

# +
#mlp_minprec_model = MLP(hidden_width=2, loss=MinPrecLoss, loss_arguments=loss_params)

if modelchosen=='mlp' and losschosen == 'minprec':
    mlp_minprec_model = MLP(hidden_width=2, loss=MinPrecLoss, loss_arguments=loss_params)
elif modelchosen == 'mlp' and losschosen == 'bce':
    mlp_minprec_model = MLP(hidden_width=2, loss=nn.BCEWithLogitsLoss, loss_arguments={})
    ymoons_train = ymoons_train[:, np.newaxis]
elif modelchosen == 'linear' and losschosen == 'bce':
    param_init = np.zeros(xmoons_train.shape[1] + 1)
    mlp_minprec_model = LinearScore(nfeat=xmoons_train.shape[1], model_param_init=param_init, loss=nn.BCEWithLogitsLoss, loss_arguments={})
elif modelchosen == 'linear' and losschosen == 'minprec':
    param_init = np.zeros(xmoons_train.shape[1] + 1)
    mlp_minprec_model = LinearScore(nfeat=xmoons_train.shape[1], model_param_init=param_init, loss=MinPrecLoss, loss_arguments=loss_params)


#mlp_minprec_model.apply(weights_init)


optimizer_minprec = optim.Adam(mlp_minprec_model.parameters(), lr=lr, weight_decay=L2_coeff)
mlp_minprec_model.fit(xmoons_train, ymoons_train, optimizer_minprec)
    
phat_minprec = mlp_minprec_model.predict_proba(xmoons_test)
#yhat = (phat >= 0.5).astype(int)
    
#prec = precision_score(ytest_toy, yhat)
#rec = recall_score(ytest_toy, yhat)

#prec, rec

np.random.choice(phat_minprec,10)
# -

decision_line = decisionLine(mlp_minprec_model, xmoons_test)
decision_line[:10]

# +
cdict = {0: 'green', 1: 'blue'}
mdict = {0:'o', 1:'x'}

fig, ax = plt.subplots()
for g in np.unique(ymoons_test):
    ix = np.where(ymoons_test == g)
    ax.scatter(xmoons_test[ix,0], xmoons_test[ix,1], c = cdict[g], marker=mdict[g], label = g, alpha=0.3)
ax.legend()

ax.plot(decision_line[:,0], decision_line[:, 1], c='red')

plt.show()

# +
yhat = (phat_minprec >= 0.5).astype(int)
    
prec = precision_score(ymoons_test, yhat)
rec = recall_score(ymoons_test, yhat)

prec, rec
# -
# ### Dataset from paper

# +
#https://github.com/tufts-ml/false-alarm-control/blob/main/toy_example_comparing_BCE_Hinge_and_Sigmoid.ipynb

from data import toydata
# -

x_toy, y_toy, _, _, _, _ = toydata.create_toy_dataset()

# +
#cdict = {0: 'green', 1: 'blue'}
#mdict = {0:'o', 1:'x'}

#fig, ax = plt.subplots()
#for g in np.unique(y_toy):
#    ix = np.where(y_toy == g)
#    ax.scatter(x_toy[ix,0], x_toy[ix,1], c = cdict[g], marker=mdict[g], label = g, alpha=0.3)
#ax.legend()

#plt.show()

# +
idx = np.array(range(y_toy.shape[0]))

rng.shuffle(idx)

# +
xtrain_toy = x_toy[idx[:300]]
ytrain_toy = y_toy[idx[:300]]

xtest_toy = x_toy[idx[300:]]
ytest_toy = y_toy[idx[300:]]

# +
#loss_params = {"pos_weight": 3 * torch.ones(len(y_basic))}

#mlp_model = MLP(hidden_width=4, loss=nn.BCEWithLogitsLoss, loss_arguments={})
if modelchosen=='mlp' and losschosen == 'minprec':
    mlp_minprec_model = MLP(hidden_width=2, loss=MinPrecLoss, loss_arguments=loss_params)
elif modelchosen == 'mlp' and losschosen == 'bce':
    mlp_minprec_model = MLP(hidden_width=2, loss=nn.BCEWithLogitsLoss, loss_arguments={})
    ytrain_toy = ytrain_toy[:, np.newaxis]
elif modelchosen == 'linear' and losschosen == 'bce':
    param_init = np.zeros(xtrain_toy.shape[1] + 1)
    mlp_minprec_model = LinearScore(nfeat=xtrain_toy.shape[1], model_param_init=param_init, loss=nn.BCEWithLogitsLoss, loss_arguments={})
elif modelchosen == 'linear' and losschosen == 'minprec':
    param_init = np.zeros(xtrain_toy.shape[1] + 1)
    mlp_minprec_model = LinearScore(nfeat=xtrain_toy.shape[1], model_param_init=param_init, loss=MinPrecLoss, loss_arguments=loss_params)


optimizer = optim.Adam(mlp_minprec_model.parameters(), lr=lr, weight_decay=L2_coeff)
mlp_minprec_model.fit(xtrain_toy, ytrain_toy, optimizer)
    
phat = mlp_minprec_model.predict_proba(xtest_toy)
#yhat = (phat >= 0.5).astype(int)
    
#prec = precision_score(ytest_toy, yhat)
#rec = recall_score(ytest_toy, yhat)

#prec, rec
np.random.choice(phat,10)
# -

decision_line = decisionLine(mlp_minprec_model, xtest_toy)
decision_line[:10]

# +
cdict = {0: 'green', 1: 'blue'}
mdict = {0:'o', 1:'x'}

fig, ax = plt.subplots()
for g in np.unique(ytest_toy):
    ix = np.where(ytest_toy == g)
    ax.scatter(xtest_toy[ix,0], xtest_toy[ix,1], c = cdict[g], marker=mdict[g], label = g, alpha=0.3)
ax.legend()

#ax.plot(decision_line[:,0], decision_line[:, 1], c='red')
ax.scatter(decision_line[:,0], decision_line[:, 1], c='red', s=0.1, alpha=0.5)

plt.show()

# +
yhat = (phat >= 0.5).astype(int)
    
prec = precision_score(ytest_toy, yhat)
rec = recall_score(ytest_toy, yhat)

prec, rec
# -

