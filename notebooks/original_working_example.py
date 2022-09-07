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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# https://www.cl.cam.ac.uk/teaching/2021/LE49/probnn/3-3.pdf

# +
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import matplotlib.pyplot as plt

# +
n = 10000

rng = np.random.default_rng(5)

#x1 = np.linspace(-10, 10, n)
#x2 = np.linspace(-10, 10, n)

x1 = rng.normal(0, 1, size=n)
x2 = rng.normal(3, 2.2, size=n)


beta0_true = -0.4
beta1_true = -5.3
beta2_true = 3.1

p = 1/(1+np.exp(-(beta0_true + beta1_true*x1 + beta2_true*x2)))


#y = np.random.binomial(1, p, size=n)
y = rng.binomial(1, p, size=n)
# -

plt.scatter(x1, x2, c=p)

# +
test = np.ones([3,3])

#newcol = np.array([1,2,3])
newcol = 2*np.ones([test.shape[1]])

np.column_stack((test, newcol))
# -

x = np.column_stack((x1, x2))

torch.from_numpy(np.ones([3,3])).shape[1]

test = nn.Parameter()

test.shape


class LogReg(nn.Module):
    def __init__(self, nfeat):
        super().__init__()
        
        self.nfeat = nfeat
        self.beta = nn.Parameter(torch.zeros([nfeat+1], dtype=torch.float))
        
    def forward(self, x, y):
        x = np.column_stack((x, np.ones([x.shape[0]])))
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y)#.type(torch.LongTensor)
        
        #self.f = torch.einsum('ij,j->i', x, self.beta)
        self.f = x @ self.beta
        
        #self.p = 1/(1+torch.exp(-self.f))
        #return y*torch.log(self.p) + (1-y)*torch.log(1-self.p)
        
        stablelogisticloss = nn.BCEWithLogitsLoss()
        return stablelogisticloss(self.f, y.float())
    
    def predict_proba(self, xtest):
        if len(xtest.shape) == 1:
            xtest = xtest.reshape([1, self.nfeat])
        
        xtest = np.column_stack((xtest, np.ones([xtest.shape[0]])))
        xtest = torch.from_numpy(xtest).float()
        
        f = xtest @ self.beta
        return torch.sigmoid(f).detach().numpy().flatten()
        #return (1/(1+torch.exp(-(self.beta0 + self.beta1*xtest)))).detach().numpy().flatten()


x[10:12,:]

np.column_stack((x[10:12,:], np.ones([x[10:12,:].shape[0]])))

x[10,:].shape

len(x[10,:].shape)

len(x[10,:].reshape([1,2]).shape)

test = x[10,:]
test = test[np.newaxis,:]

np.column_stack((test, np.ones(test.shape[0])))



np.column_stack((x[10,:], np.ones([x[10,:].shape[0]])))

model = LogReg(nfeat=x.shape[1])

optimizer = optim.Adam(model.parameters(), lr=0.1)

for _ in range(1000):
    optimizer.zero_grad()
    loss = model(x, y)
    #mean_loss = -torch.mean(loss)
    #mean_loss.backward()
    loss.backward()
    optimizer.step()

loss

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

(beta1_true, beta2_true, beta0_true)

model.predict_proba(x[696,:])

p[696]

# +
ntest = int(np.floor(n/10))

x1test = rng.normal(0, 1, size=ntest)
x2test = rng.normal(3, 2.2, size=ntest)

ptest = 1/(1+np.exp(-(beta0_true + beta1_true*x1test + beta2_true*x2test)))

xtest = np.column_stack((x1test, x2test))

ytest = rng.binomial(1, ptest, size=ntest)

# +
thresh = 0.5

phat = model.predict_proba(xtest)
yhat = (phat >= thresh).astype(int)

# +
from sklearn.metrics import precision_score, recall_score

prec = precision_score(ytest, yhat)
rec = recall_score(ytest, yhat)

(prec, rec)
# -
# Use logistic regression to set initialisation values, since non-convexity of problem otherwise might be the cause of beta1 being zero and the decision boundary being non-sensical (since beta values currently initialized to 0).


(beta1_true, beta2_true, beta0_true)

# The version of the model below initialised its parameters to zero, which given that the loss is non-convex, resulted in sometimes parameters staying at zero.  
#
# Modified this model below to initialise using the logistic regression best fit values, which should be a much more sensible starting point.

# class LogRegMaxRecall(nn.Module):
#     def __init__(self, nfeat, min_prec=0.8, lam=1e4, lr=1e-3):
#         super().__init__()
#         #self.beta0 = nn.Parameter(torch.tensor(0, dtype=torch.float))
#         #self.beta1 = nn.Parameter(torch.tensor(1, dtype=torch.float))
#         self.nfeat = nfeat
#         self.min_prec = min_prec
#         self.beta = nn.Parameter(torch.zeros([nfeat+1], dtype=torch.float))
#         
#         #Page 6 of paper, second-to-last paragraph
#         self.lmbda = lam
#                 
#     def forward(self, x, y):
#         x = np.column_stack((x, np.ones([x.shape[0]])))
#         x = torch.from_numpy(x).float()
#         y = torch.from_numpy(y)#.type(torch.LongTensor)
#         
#         #Paragraph below eqn 14 in paper
#         self.gamma = torch.tensor(7.)
#         self.delta = torch.tensor(0.035)
#         
#         #self.f = self.beta0 + self.beta1*x
#         self.f = x @ self.beta
#         
#         #self.p = 1/(1+torch.exp(-self.f))
#         
#         #Eqn 11 in paper
#         tpc = torch.sum(torch.where(y==1., self.gamma*self.delta + torch.where(self.f>0., torch.tensor(1.), torch.tensor(0.)), torch.tensor(0.)))
#         #tpc.requires_grad = True
#         #Why the torch.tensor() on the scalars is needed:
#         #https://github.com/pytorch/pytorch/issues/9190#issuecomment-402837158
#         
#         #print(tpc)
#         
#         #Paragraph below eqn. 10
#         self.mhat = 6.85
#         self.bhat = 1.59
#         
#         #Line below eqn. 1 in paper
#         self.Nplus = torch.sum(y)
#         
#         #Eqn 10
#         fpc = torch.sum(torch.where(y==0., (1+self.gamma*self.delta)*torch.sigmoid(self.mhat*self.f+self.bhat), torch.tensor(0.)))
#         
#         g = -tpc + self.min_prec/(1.-self.min_prec)*fpc + self.gamma*self.delta*self.Nplus
#         
#         #logitloss = y*torch.log(self.p) + (1-y)*torch.log(1-self.p)
#         #stablelogisticloss = nn.BCEWithLogitsLoss()
#         #return stablelogisticloss(self.f, y.float())
#         
#         #Eqn. 12
#         loss = -tpc + self.lmbda*nn.ReLU()(g)
#         #The reason for the odd way of calling the ReLU function:
#         #https://discuss.pytorch.org/t/multiplication-of-activation-function-with-learnable-parameter-scalar/113746/2
#         
#         return loss
#     
#     def predict_proba(self, xtest):
#         if len(xtest.shape) == 1:
#             xtest = xtest.reshape([1, self.nfeat])
#         
#         xtest = np.column_stack((xtest, np.ones([xtest.shape[0]])))
#         xtest = torch.from_numpy(xtest).float()
#         
#         f = xtest @ self.beta
#         return torch.sigmoid(f).detach().numpy().flatten()
#         #xtest = torch.from_numpy(xtest)
#         #return (1/(1+torch.exp(-(self.beta0 + self.beta1*xtest)))).detach().numpy().flatten()

# +
from sklearn.linear_model import LogisticRegression

sklearnlogreg = LogisticRegression()
sklearnlogreg = sklearnlogreg.fit(x,y)
sklearnbetas = np.concatenate([sklearnlogreg.coef_.flatten(), sklearnlogreg.intercept_])
sklearnbetas


# -

class LogRegMaxRecall(nn.Module):
    def __init__(self, x, y, min_prec=0.8, lam=1e4):
        super().__init__()
        #self.beta0 = nn.Parameter(torch.tensor(0, dtype=torch.float))
        #self.beta1 = nn.Parameter(torch.tensor(1, dtype=torch.float))
        
        #self.nfeat = nfeat
        self.min_prec = min_prec
        
        #self.beta = nn.Parameter(torch.zeros([nfeat+1], dtype=torch.float))
        sklearnlogreg = LogisticRegression()
        sklearnlogreg = sklearnlogreg.fit(x,y)
        sklearnbetas = np.concatenate([sklearnlogreg.coef_.flatten(), sklearnlogreg.intercept_])
        
        self.beta = nn.Parameter(torch.from_numpy(sklearnbetas).float())

        #Page 6 of paper, second-to-last paragraph
        self.lmbda = lam
                
    def forward(self, x, y):
        x = np.column_stack((x, np.ones([x.shape[0]])))
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y)#.type(torch.LongTensor)
        
        #Paragraph below eqn 14 in paper
        self.gamma = torch.tensor(7.)
        self.delta = torch.tensor(0.035)
        
        #self.f = self.beta0 + self.beta1*x
        self.f = x @ self.beta
        
        #self.p = 1/(1+torch.exp(-self.f))
        
        #Eqn 11 in paper
        #tpc = torch.sum(torch.where(y==1., self.gamma*self.delta + torch.where(self.f>0., torch.tensor(1.), torch.tensor(0.)), torch.tensor(0.)))
        #Why the torch.tensor() on the scalars is needed:
        #https://github.com/pytorch/pytorch/issues/9190#issuecomment-402837158
        
        self.mtilde = 6.85
        self.btilde = -3.54
        
        #Eqn. 14
        tpc = torch.sum(torch.where(y==1., (1+self.gamma*self.delta)*torch.sigmoid(self.mtilde*self.f+self.btilde), torch.tensor(0.)))
        
        #print(tpc)
        
        #Paragraph below eqn. 10
        self.mhat = 6.85
        self.bhat = 1.59
        
        #Eqn 10
        fpc = torch.sum(torch.where(y==0., (1+self.gamma*self.delta)*torch.sigmoid(self.mhat*self.f+self.bhat), torch.tensor(0.)))
        
        #Line below eqn. 1 in paper
        self.Nplus = torch.sum(y)
        
        #Eqn. 12
        g = -tpc + self.min_prec/(1.-self.min_prec)*fpc + self.gamma*self.delta*self.Nplus
        
        #logitloss = y*torch.log(self.p) + (1-y)*torch.log(1-self.p)
        #stablelogisticloss = nn.BCEWithLogitsLoss()
        #return stablelogisticloss(self.f, y.float())
        
        #Eqn. 12
        loss = -tpc + self.lmbda*nn.ReLU()(g)
        #The reason for the odd way of calling the ReLU function:
        #https://discuss.pytorch.org/t/multiplication-of-activation-function-with-learnable-parameter-scalar/113746/2
        
        return loss
    
    def predict_proba(self, xtest):
        if len(xtest.shape) == 1:
            xtest = xtest.reshape([1, x.shape[0]])
        
        xtest = np.column_stack((xtest, np.ones([xtest.shape[0]])))
        xtest = torch.from_numpy(xtest).float()
        
        f = xtest @ self.beta
        return torch.sigmoid(f).detach().numpy().flatten()
        #xtest = torch.from_numpy(xtest)
        #return (1/(1+torch.exp(-(self.beta0 + self.beta1*xtest)))).detach().numpy().flatten()


# +
model2 = LogRegMaxRecall(x, y, min_prec=0.8)

optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)
# -

for _ in range(100):
    optimizer2.zero_grad()
    loss2 = model2(x, y)
    #loss2 = Variable(loss2, requires_grad=True)
    #mean_loss2 = -torch.mean(loss2)
    #mean_loss2.backward()
    loss2.backward()
    optimizer2.step()

for name, param in model2.named_parameters():
    if param.requires_grad:
        print(name, param.data)

# +
thresh2 = 0.5

phat2 = model2.predict_proba(xtest)
yhat2 = (phat2 >= thresh2).astype(int)

# +
from sklearn.metrics import precision_score, recall_score

prec2 = precision_score(ytest, yhat2)
rec2 = recall_score(ytest, yhat2)

(prec2, rec2)
# -

loss

# Fitting a perfectly specified model is not a typical use case of this algorithm.  
# Let's try different datasets.

# ### A 1D dataset

# +
from sklearn.datasets import make_classification

xsk, ysk = make_classification(n_samples=10000, n_features=1, n_informative=1, n_redundant=0, n_classes=2, n_clusters_per_class=1, class_sep=2., random_state=6)
# -

ysk.dtype

ysk[:10]

plt.scatter(xsk, ysk)

# +
xsk_train = xsk[:9000]
ysk_train = ysk[:9000]

xsk_test = xsk[9000:]
ysk_test = ysk[9000:]
# -


ysk_train

# +
model = LogReg(nfeat=xsk_train.shape[1])

optimizer = optim.Adam(model.parameters(), lr=0.1)
# -

for _ in range(1000):
    optimizer.zero_grad()
    loss = model(xsk_train, ysk_train)
    #mean_loss = -torch.mean(loss)
    #mean_loss.backward()
    loss.backward()
    optimizer.step()

loss

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

psk_hat = model.predict_proba(xsk_test)

# +
thresh = 0.5

ysk_hat = (psk_hat > thresh).astype(int)
ysk_hat[:10]

# +
prec = precision_score(ysk_test, ysk_hat)
rec = recall_score(ysk_test, ysk_hat)

(prec, rec)
# -
(ysk_hat != ysk_test).sum()

# Precision and recall scores are identical. Possibly fine, but slightly suspicious. Manually check them.

((ysk_hat==1) & (ysk_hat==ysk_test)).sum()/len(ysk_hat[ysk_hat==1])

((ysk_hat==1) & (ysk_hat==ysk_test)).sum()/len(ysk_test[ysk_test==1])

# They are the same because the denominators are equal: there are the same number of predicted positives as there are of total positives. (And the numerator is the same in both by definition: number of true positives).



# +
#model2 = LogRegMaxRecall(nfeat=xsk_train.shape[1], min_prec=0.8)
model2 = LogRegMaxRecall(xsk_train, ysk_train, min_prec=0.8)

optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)
# -

for _ in range(100):
    optimizer2.zero_grad()
    loss2 = model2(xsk_train, ysk_train)
    #mean_loss2 = -torch.mean(loss2)
    #mean_loss2.backward()
    loss2.backward()
    optimizer2.step()

loss2

for name, param in model2.named_parameters():
    if param.requires_grad:
        print(name, param.data)

# +
thresh2 = 0.5

phat2 = model2.predict_proba(xsk_test)
yhat2 = (phat2 >= thresh2).astype(int)

# +
from sklearn.metrics import precision_score, recall_score

prec2 = precision_score(ysk_test, yhat2)
rec2 = recall_score(ysk_test, yhat2)

(prec2, rec2)
# -


# ### Moons dataset


# +
from sklearn.datasets import make_moons

xmoons, ymoons = make_moons(n_samples = 10000, noise=0.1, random_state=7)

# +
xmoons_train = xmoons[:9000]
ymoons_train = ymoons[:9000]

xmoons_test = xmoons[9000:]
ymoons_test = ymoons[9000:]
# -

plt.scatter(xmoons_train[:,0], xmoons_train[:,1], c=ymoons_train, alpha=0.3)

# To plot decision boundary solve this equation for x, and evaluate it at p=0.5 (the chosen decision threshold):  
# $$p = \frac{1}{1+\exp({-x\beta})}$$

# $$x\beta = -\ln\left( \frac{1}{p} - 1\right)$$
# (Note that for $p=0.5$ this gives $x\beta = 0$)



model = LogReg(nfeat=xmoons_train.shape[1])

optimizer = optim.Adam(model.parameters(), lr=0.1)

for _ in range(1000):
    optimizer.zero_grad()
    loss = model(xmoons_train, ymoons_train)
    #mean_loss = -torch.mean(loss)
    #mean_loss.backward()
    loss.backward()
    optimizer.step()

loss

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

phat = model.predict_proba(xmoons_test)

yhat = (phat >= thresh).astype(int)
yhat[:10]

np.unique(yhat, return_counts=True)

betas = [i[1].data.numpy() for i in model.named_parameters() if i[1].requires_grad]
betas = betas[0]
betas

# +
p_thresh = 0.5

x1 = np.linspace(np.min(xmoons), np.max(xmoons), 1000)
x2 = (-np.log(1/p_thresh - 1) -betas[0]*x1 - betas[2])/betas[1]

# +
#fig, ax = plt.subplots()

#ax.scatter(xmoons_test[:,0], xmoons_test[:,1], c=ymoons_test, alpha=0.3)
#ax.plot(x1, x2, c='r')

#plt.plot()

# +
cdict = {0: 'green', 1: 'blue'}
mdict = {0:'o', 1:'x'}

fig, ax = plt.subplots()
for g in np.unique(ymoons_test):
    ix = np.where(ymoons_test == g)
    ax.scatter(xmoons_test[ix,0], xmoons_test[ix,1], c = cdict[g], marker=mdict[g], label = g, alpha=0.3)
ax.legend()

ax.plot(x1, x2, c='r')

plt.show()

# +
thresh = 0.5

ymoons_hat = (phat > thresh).astype(int)
ymoons_hat[:10]

# +
prec = precision_score(ymoons_test, ymoons_hat)
rec = recall_score(ymoons_test, ymoons_hat)

(prec, rec)
# -






# +
#model2 = LogRegMaxRecall(nfeat=xmoons_train.shape[1], min_prec=0.5, lam=10000)
model2 = LogRegMaxRecall(xmoons_train, ymoons_train, min_prec=0.9, lam=1000)

optimizer2 = optim.Adam(model2.parameters(), lr=1e-1)
# -

for _ in range(1000):
    optimizer2.zero_grad()
    loss2 = model2(xmoons_train, ymoons_train)
    #mean_loss2 = -torch.mean(loss2)
    #mean_loss2.backward()
    loss2.backward()
    optimizer2.step()

loss2

for name, param in model2.named_parameters():
    if param.requires_grad:
        print(name, param.data)

betas = [i[1].data.numpy() for i in model2.named_parameters() if i[1].requires_grad]
betas = betas[0]
betas

# +
p_thresh = 0.5

x1 = np.linspace(np.min(xmoons), np.max(xmoons), 1000)
x2 = (-np.log(1/p_thresh - 1) -betas[0]*x1 - betas[2])/betas[1]

# +
#fig, ax = plt.subplots()

#ax.scatter(xmoons_test[:,0], xmoons_test[:,1], c=ymoons_test, alpha=0.3)
#ax.plot(x1, x2, c='r')
#ax.legend()

#plt.plot()

# +
cdict = {0: 'green', 1: 'blue'}
mdict = {0:'o', 1:'x'}

fig, ax = plt.subplots()
for g in np.unique(ymoons_test):
    ix = np.where(ymoons_test == g)
    ax.scatter(xmoons_test[ix,0], xmoons_test[ix,1], c = cdict[g], marker=mdict[g], label = g, alpha=0.3)
ax.legend()

ax.plot(x1, x2, c='r')

plt.show()

# +
thresh2 = 0.5

phat2 = model2.predict_proba(xmoons_test)
yhat2 = (phat2 >= thresh2).astype(int)

# +
from sklearn.metrics import precision_score, recall_score

prec2 = precision_score(ymoons_test, yhat2)
rec2 = recall_score(ymoons_test, yhat2)

(prec2, rec2)
# -
# (0.9854227405247813, 0.7115789473684211)







# ### Dataset from paper

# +
#https://github.com/tufts-ml/false-alarm-control/blob/main/toy_example_comparing_BCE_Hinge_and_Sigmoid.ipynb

import toydata

x_toy, y_toy, _, _, _, _ = toydata.create_toy_dataset()
# -

x_toy.shape

y_toy.shape

# +
cdict = {0: 'green', 1: 'blue'}
mdict = {0:'o', 1:'x'}

fig, ax = plt.subplots()
for g in np.unique(y_toy):
    ix = np.where(y_toy == g)
    ax.scatter(x_toy[ix,0], x_toy[ix,1], c = cdict[g], marker=mdict[g], label = g, alpha=0.3)
ax.legend()

#ax.plot(x1, x2, c='r')

plt.show()
# -

y_toy[:200]

# Need to shuffle the data before doing train/test split!

# +
idx = np.array(range(y_toy.shape[0]))

rng.shuffle(idx)
# -

idx[:10]

# +
xtrain_toy = x_toy[idx[:300]]
ytrain_toy = y_toy[idx[:300]]

xtest_toy = x_toy[idx[300:]]
ytest_toy = y_toy[idx[300:]]
# -



# +
model = LogReg(nfeat=xtrain_toy.shape[1])

optimizer = optim.Adam(model.parameters(), lr=0.1)
# -

for _ in range(1000):
    optimizer.zero_grad()
    loss = model(xtrain_toy, ytrain_toy)
    #mean_loss = -torch.mean(loss)
    #mean_loss.backward()
    loss.backward()
    optimizer.step()

loss

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

psk_hat = model.predict_proba(xtest_toy)

betas = [i[1].data.numpy() for i in model.named_parameters() if i[1].requires_grad]
betas = betas[0]
betas

# +
p_thresh = 0.5

x1 = np.linspace(np.percentile(xtest_toy, 0), np.percentile(xtest_toy, 95), 1000)
x2 = (-np.log(1/p_thresh - 1) -betas[0]*x1 - betas[2])/betas[1]

# +
cdict = {0: 'green', 1: 'blue'}
mdict = {0:'o', 1:'x'}

fig, ax = plt.subplots()
for g in np.unique(ytest_toy):
    ix = np.where(ytest_toy == g)
    ax.scatter(xtest_toy[ix,0], xtest_toy[ix,1], c = cdict[g], marker=mdict[g], label = g, alpha=0.3)
ax.legend()

ax.plot(x1, x2, c='r')

plt.show()

# +
thresh = 0.5

ysk_hat = (psk_hat > thresh).astype(int)
ysk_hat[:10]

# +
prec = precision_score(ytest_toy, ysk_hat)
rec = recall_score(ytest_toy, ysk_hat)

(prec, rec)
# -




# +
#model2 = LogRegMaxRecall(nfeat=xtrain_toy.shape[1], min_prec=0.8, lam=10000)
model2 = LogRegMaxRecall(xtrain_toy, ytrain_toy, min_prec=0.85, lam=10000)

optimizer2 = optim.Adam(model2.parameters(), lr=1e-3)
# -

for _ in range(1000):
    optimizer2.zero_grad()
    loss2 = model2(xtrain_toy, ytrain_toy)
    #mean_loss2 = -torch.mean(loss2)
    #mean_loss2.backward()
    loss2.backward()
    optimizer2.step()

loss2

for name, param in model2.named_parameters():
    if param.requires_grad:
        print(name, param.data)

betas = [i[1].data.numpy() for i in model2.named_parameters() if i[1].requires_grad]
betas = betas[0]
betas

# +
p_thresh = 0.5

x1 = np.linspace(np.percentile(xtest_toy, 0), np.percentile(xtest_toy, 95), 1000)
x2 = (-np.log(1/p_thresh - 1) -betas[0]*x1 - betas[2])/betas[1]

# +
cdict = {0: 'green', 1: 'blue'}
mdict = {0:'o', 1:'x'}

fig, ax = plt.subplots()
for g in np.unique(ytest_toy):
    ix = np.where(ytest_toy == g)
    ax.scatter(xtest_toy[ix,0], xtest_toy[ix,1], c = cdict[g], marker=mdict[g], label = g, alpha=0.3)
ax.legend()

ax.plot(x1, x2, c='r')

plt.show()

# +
thresh2 = 0.5

phat2 = model2.predict_proba(xtest_toy)
yhat2 = (phat2 >= thresh2).astype(int)

# +
from sklearn.metrics import precision_score, recall_score

prec2 = precision_score(ytest_toy, yhat2)
rec2 = recall_score(ytest_toy, yhat2)

(prec2, rec2)
# -

