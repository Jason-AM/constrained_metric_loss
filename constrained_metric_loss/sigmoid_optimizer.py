import torch
import torch.nn as nn


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
