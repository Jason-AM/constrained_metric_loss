from scipy.special import expit
from scipy.optimize import minimize
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class tight_sigmoid(nn.Module):
    def __init__(self, gamma, delta, eps, upper_or_lower):
        super().__init__()

        self.eps = eps
        self.delta = delta
        self.gamma = gamma
        self.upper_or_lower = upper_or_lower

        self.m = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))

    def upper_bound_obj(self):
        return torch.sum(
            torch.square(self.delta - self.general_sigmoiod(-self.eps))
            + torch.square(1 + self.delta - self.general_sigmoiod(0))
        )

    def lower_bound_obj(self):
        return torch.sum(
            torch.square(1 + self.delta - self.general_sigmoiod(self.eps))
            + torch.square(self.delta - self.general_sigmoiod(0))
        )

    def general_sigmoiod(self, a):
        return (1 + self.gamma * self.delta) * torch.sigmoid(self.m * a + self.b)

    def forward(self):
        if self.upper_or_lower == "upper":
            return self.upper_bound_obj()
        elif self.upper_or_lower == "lower":
            return self.lower_bound_obj()
        else:
            raise "Need to select upper or lower bound using 'upper' or 'lower'"


def get_sigmoid_params_pytorch(gamma, delta, eps, upper_or_lower, epochs=1000):
    model = tight_sigmoid(gamma, delta, eps, upper_or_lower)
    optimizer = optim.Adam(model.parameters(), lr=1.0)

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()

    m, b = model.named_parameters()
    m = float(m[1].detach().numpy())
    b = float(b[1].detach().numpy())
    return m, b


def get_sigmoid_params_bfgs(gamma, delta, eps, upper_or_lower):
    def _general_sigmoiod(x, gamma, delta, m, b):
        return (1 + gamma * delta) * expit(m * x + b)

    def _lower_bound(params_to_minimise, gamma, delta, eps):
        m, b = params_to_minimise
        return np.sum(
            np.square(1 + delta - _general_sigmoiod(eps, gamma, delta, m, b))
            + np.square(delta - _general_sigmoiod(0, gamma, delta, m, b))
        )

    def _upper_bound(params_to_minimise, gamma, delta, eps):
        m, b = params_to_minimise
        return np.sum(
            np.square(delta - _general_sigmoiod(-eps, gamma, delta, m, b))
            + np.square(1 + delta - _general_sigmoiod(0, gamma, delta, m, b))
        )

    if upper_or_lower == "lower":
        return minimize(_lower_bound, x0=(1, 1), args=(gamma, delta, eps), method="BFGS").x

    elif upper_or_lower == "upper":
        return minimize(_upper_bound, x0=(1, 1), args=(gamma, delta, eps), method="BFGS").x
