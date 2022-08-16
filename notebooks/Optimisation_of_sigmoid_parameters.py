# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# import cvxpy as cp
#
# def sigmoid(x):
#     return cp.inv_pos(1+cp.exp(-x)) #https://stackoverflow.com/a/71644683
#
# eps = 0.75
# delta = 0.035
# gamma = 7
#
# m = cp.Variable(1)
# b = cp.Variable(1)
#
# #Eqn 9
# def u(a):
#     return (1+gamma*delta)*sigmoid(m*a + b)
#
# objective = cp.Minimize(cp.sum(cp.square(delta - u(-eps)) + cp.square(1 + delta - u(0))))
#
# #constraints = []
# prob = cp.Problem(objective)
#
# result = prob.solve()
#
# print((m.value, b.value))
# https://github.com/cvxpy/cvxpy/discussions/1468#discussioncomment-1278396

# from analyzer import tech_support
#
# tech_support(prob)


# +
import torch
import torch.nn as nn
import torch.optim as optim

class tight_sigmoid(nn.Module):
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
        
        # obj = torch.sum(torch.square(self.delta - u(-self.eps)) + torch.square(1 + self.delta - u(0)))
        obj = torch.sum(torch.square(1 + self.delta - u(self.eps)) + torch.square(self.delta - u(0)))
        return obj


# -

eps = 0.75
delta = 0.035
gamma = 7

model = tight_sigmoid(eps, delta, gamma)

optimizer = optim.Adam(model.parameters(), lr=1)
#optimizer = optim.LBFGS(model.parameters())

for _ in range(1000):
    optimizer.zero_grad()
    loss = model()
    loss.backward()
    optimizer.step()

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

# +
m, b = model.named_parameters()
m = float(m[1].detach().numpy())
b = float(b[1].detach().numpy())

print((m,b))
# -

# Fitted values shown in paper (below eqn. 10) are:  
# "m = 6.85 and shift b = 1.59"

# +
import numpy as np

def u(a):
    return (1+gamma*delta)/(1 + np.exp(-(m*a + b)))


# +
x = np.linspace(-1, 1, 100)

y = u(x)

# +
import matplotlib.pyplot as plt

plt.axhline(delta, color='r', linestyle='--')
plt.axvline(-eps, color='r', linestyle='--')

plt.axhline(1+delta, color='g', linestyle='--')
plt.axvline(0, color='g', linestyle='--')

plt.plot(x, y)
# -

# Checking if fitted values satisfy the constraints they should (eqn. 8 in paper).  
# Each of the two consecutive quantities below should be approximately equal.

u(-eps)

delta



u(0)

1+delta



# Next step: plot the loss as a function of m,b. Purely to see what it looks like.


