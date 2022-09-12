import torch
import torch.nn as nn


class BCELogitManual(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f, y):
        def _logsumexp(f):
            return torch.logsumexp(torch.column_stack((torch.zeros(len(f), 1), (-f).reshape(-1, 1))), 1)

        def zero_one_loss(y, f):
            return 1 - torch.heaviside((2 * y - 1) * f, torch.tensor(0.0))

        def sigmoid_approx_of_zero_one_loss(y, f, m, b):
            return 1 - torch.sigmoid((2 * y - 1) * (m * f + b))

        def logloss_approx_of_zero_one(y, f):
            return -(y * torch.log(torch.sigmoid(f)) + (1 - y) * torch.log(1 - torch.sigmoid(f)))

        def relu_of_leaky_relu(y, f):
            return 1.0 - nn.ReLU()(1.0 - nn.LeakyReLU(negative_slope=0.01)((2 * y - 1) * f))

        # tpc = torch.sum(torch.where(y == 1.0, torch.log(torch.tensor(1)) - _logsumexp(f), torch.tensor(0.0)))

        # tnc = torch.sum(torch.where(y == 0.0, torch.log(torch.tensor(1)) - _logsumexp(1 - f), torch.tensor(0.0)))

        # tpc = torch.sum(torch.where(y == 1.0, 1 - logloss_approx_of_zero_one(1, f), torch.tensor(0.0)))

        # tnc = torch.sum(torch.where(y == 0.0, 1 - logloss_approx_of_zero_one(0, f), torch.tensor(0.0)))

        tpc = torch.sum(torch.where(y == 1.0, 1 - relu_of_leaky_relu(1, f), torch.tensor(0.0)))

        tnc = torch.sum(torch.where(y == 0.0, 1 - relu_of_leaky_relu(0, f), torch.tensor(0.0)))

        # tpc = torch.sum(torch.where(y == 1.0, 1 - sigmoid_approx_of_zero_one_loss(1, f, 1, 3), torch.tensor(0.0)))

        # tnc = torch.sum(torch.where(y == 0.0, 1 - sigmoid_approx_of_zero_one_loss(0, f, 1, -3), torch.tensor(0.0)))

        # tpc = torch.sum(torch.where(y == 1.0, 1 - zero_one_loss(1, f), torch.tensor(0.0)))

        # tnc = torch.sum(torch.where(y == 0.0, 1 - zero_one_loss(0, f), torch.tensor(0.0)))

        return -tpc - tnc
