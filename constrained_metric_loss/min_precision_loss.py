from typing import Dict, Optional


import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss

from constrained_metric_loss.sigmoid_optimizer import get_sigmoid_params_bfgs


class MinPrecLoss(nn.Module):
    def __init__(
        self,
        min_prec: float,
        lmbda: float,
        sigmoid_hyperparams: Dict,
        sigmoid_params: Optional[Dict] = None,
    ):
        super().__init__()
        self.min_prec = min_prec
        self.lmbda = lmbda

        self.extract_sigmoid_hyperparameters(sigmoid_hyperparams)
        self.extract_or_calculate_sigmoid_params(sigmoid_params)

    def extract_sigmoid_hyperparameters(self, sigmoid_hyperparams: Dict):
        self.gamma = sigmoid_hyperparams["gamma"]
        self.delta = sigmoid_hyperparams["delta"]
        self.eps = sigmoid_hyperparams.get("eps", None)

    def extract_or_calculate_sigmoid_params(self, sigmoid_params: Optional[Dict]):

        if sigmoid_params is None:
            self.mtilde, self.btilde = get_sigmoid_params_bfgs(self.gamma, self.delta, self.eps, "lower")
            self.mhat, self.bhat = get_sigmoid_params_bfgs(self.gamma, self.delta, self.eps, "upper")
        else:
            self.mtilde = sigmoid_params["mtilde"]
            self.btilde = sigmoid_params["btilde"]
            self.mhat = sigmoid_params["mhat"]
            self.bhat = sigmoid_params["bhat"]

    def forward(self, f, y):
        y = y.flatten()
        f = f.flatten()

        tpc = torch.sum(
            torch.where(
                y == 1.0,
                (1 + self.gamma * self.delta) * torch.sigmoid(self.mtilde * f + self.btilde),
                torch.tensor(0.0),
            )
        )

        # tpc = torch.dot(
        #     y.flatten(), (1 + self.gamma * self.delta) * torch.sigmoid(self.mtilde * f + self.btilde).flatten()
        # )

        # Eqn 10
        fpc = torch.sum(
            torch.where(
                y == 0.0, (1 + self.gamma * self.delta) * torch.sigmoid(self.mhat * f + self.bhat), torch.tensor(0.0)
            )
        )

        # fpc = torch.dot(
        #     (1 - y).flatten(), (1 + self.gamma * self.delta) * torch.sigmoid(self.mhat * f + self.bhat).flatten()
        # )

        # Line below eqn. 1 in paper
        Nplus = torch.sum(y)

        # Eqn. 12
        g = -tpc + (self.min_prec / (1.0 - self.min_prec)) * fpc + self.gamma * self.delta * Nplus

        # Eqn. 12
        loss = -tpc + self.lmbda * nn.ReLU()(g)
        # The reason for the odd way of calling the ReLU function:
        # https://discuss.pytorch.org/t/multiplication-of-activation-function-with-learnable-parameter-scalar/113746/2

        # print(tpc, fpc, loss, g, self.lmbda, self.min_prec)
        return loss


class LearnableMinPrecLoss(nn.Module):
    def __init__(
        self,
        min_prec: float,
        lmbda: float,
        lmbda2: float,
        sigmoid_hyperparams: Dict,
    ):
        super().__init__()
        self.min_prec = min_prec
        self.lmbda = lmbda
        self.lmbda2 = lmbda2

        self.sigmoid_scale = sigmoid_hyperparams["sigmoid_scale"]
        self.mtilde = sigmoid_hyperparams["mtilde"]
        self.btilde = sigmoid_hyperparams["btilde"]
        self.mhat = sigmoid_hyperparams["mhat"]
        self.bhat = sigmoid_hyperparams["bhat"]

    def forward(self, f, y):
        tpc = torch.sum(
            torch.where(
                y == 1.0,
                (1 + self.sigmoid_scale) * torch.sigmoid(self.mtilde * f + self.btilde),
                torch.tensor(0.0),
            )
        )

        # Eqn 10
        fpc = torch.sum(
            torch.where(
                y == 0.0, (1 + self.sigmoid_scale) * torch.sigmoid(self.mhat * f + self.bhat), torch.tensor(0.0)
            )
        )

        # Line below eqn. 1 in paper
        Nplus = torch.sum(y)

        # Eqn. 12
        g = -tpc + self.min_prec / (1.0 - self.min_prec) * fpc + self.sigmoid_scale * Nplus

        # postive_val
        pos = -tpc - fpc

        # Eqn. 12
        loss = -tpc + self.lmbda * nn.ReLU()(g) + self.lmbda2 * nn.ReLU()(pos)
        # The reason for the odd way of calling the ReLU function:
        # https://discuss.pytorch.org/t/multiplication-of-activation-function-with-learnable-parameter-scalar/113746/2

        return loss


class MinPrecLossLogForm(nn.Module):
    def __init__(
        self,
        min_prec: float,
        lmbda: float,
        sigmoid_hyperparams: Dict,
        sigmoid_params: Optional[Dict] = None,
    ):
        super().__init__()
        self.min_prec = min_prec
        self.lmbda = lmbda

        self.extract_sigmoid_hyperparameters(sigmoid_hyperparams)
        self.extract_or_calculate_sigmoid_params(sigmoid_params)

    def extract_sigmoid_hyperparameters(self, sigmoid_hyperparams: Dict):
        self.gamma = sigmoid_hyperparams["gamma"]
        self.delta = sigmoid_hyperparams["delta"]
        self.eps = sigmoid_hyperparams.get("eps", None)

    def extract_or_calculate_sigmoid_params(self, sigmoid_params: Optional[Dict]):

        if sigmoid_params is None:
            self.mtilde, self.btilde = get_sigmoid_params_bfgs(self.gamma, self.delta, self.eps, "lower")
            self.mhat, self.bhat = get_sigmoid_params_bfgs(self.gamma, self.delta, self.eps, "upper")
        else:
            self.mtilde = sigmoid_params["mtilde"]
            self.btilde = sigmoid_params["btilde"]
            self.mhat = sigmoid_params["mhat"]
            self.bhat = sigmoid_params["bhat"]

    def forward(self, f, y):
        def _logsumexp(m, b, f):
            return torch.logsumexp(torch.column_stack((torch.zeros(len(f), 1), (-m * f - b).reshape(-1, 1))), 1)

        tpc = torch.sum(
            torch.where(
                y == 1.0,
                (torch.log(torch.tensor(1 + self.gamma * self.delta)) - _logsumexp(self.mtilde, self.btilde, f)),
                torch.tensor(0.0),
            )
        )

        fpc = torch.sum(
            torch.where(
                y == 0.0,
                (torch.log(torch.tensor(1 + self.gamma * self.delta)) - _logsumexp(self.mhat, self.bhat, f)),
                torch.tensor(0.0),
            )
        )

        # tpc = torch.sum(
        #     torch.where(
        #         y == 1.0,
        #         torch.log((1 + self.gamma * self.delta) * torch.sigmoid(self.mtilde * f + self.btilde) + 1),
        #         torch.tensor(0.0),
        #     )
        # )

        # # Eqn 10
        # fpc = torch.sum(
        #     torch.where(
        #         y == 0.0,
        #         torch.log((1 + self.gamma * self.delta) * torch.sigmoid(self.mhat * f + self.bhat) + 1e-1),
        #         torch.tensor(0.0),
        #     )
        # )

        # Line below eqn. 1 in paper
        Nplus = torch.sum(y)

        # Eqn. 12
        g = -tpc + self.min_prec / (1.0 - self.min_prec) * fpc + self.gamma * self.delta * Nplus

        # Eqn. 12
        loss = -tpc + self.lmbda * nn.ReLU()(g)  # g
        # The reason for the odd way of calling the ReLU function:
        # https://discuss.pytorch.org/t/multiplication-of-activation-function-with-learnable-parameter-scalar/113746/2

        return loss


class MinPrecLeakyLoss(_Loss):  # (nn.Module):
    def __init__(self, min_prec: float, lmbda: float, lmbda2: float, leaky_slope: float = 0.01):
        super().__init__()
        self.min_prec = min_prec
        self.lmbda = lmbda
        self.lmbda2 = lmbda2
        self.leaky_slope = leaky_slope

    @staticmethod
    def leaky_relu_01_loss(x, y, negative_slope):
        return nn.ReLU()(1 - nn.LeakyReLU(negative_slope=negative_slope)((2 * y - 1) * x))
        # return nn.LeakyReLU(negative_slope=-0.001)(1 - nn.LeakyReLU(negative_slope=negative_slope)((2 * y - 1) * x))
        # return nn.ReLU()(1 - (2 * y - 1) * x)

    @staticmethod
    def logsigmoidloss(x, y, negative_slope):
        return -torch.log(torch.sigmoid((2 * y - 1) * x))

    def forward(self, f, y):

        # tpc = torch.sum(torch.where(y == 1.0, 1 - self.leaky_relu_01_loss(f, 1.0, self.leaky_slope), torch.tensor(0.0)))
        # tpc = torch.dot(y.flatten(), 1 - self.leaky_relu_01_loss(f, 1.0, self.leaky_slope).flatten())
        # tpc = torch.dot(y.flatten(), 1 - self.logsigmoidloss(f, 1.0, self.leaky_slope).flatten())

        # fpc = torch.sum(torch.where(y == 0.0, self.leaky_relu_01_loss(f, 0.0, self.leaky_slope), torch.tensor(0.0)))
        # fpc = torch.dot(1 - y.flatten(), self.leaky_relu_01_loss(f, 0.0, self.leaky_slope).flatten())
        # fpc = torch.dot(1 - y.flatten(), self.logsigmoidloss(f, 0.0, self.leaky_slope).flatten())

        Nplus = torch.sum(y)

        # g = -tpc + self.min_prec / (1.0 - self.min_prec) * fpc + Nplus

        # loss = -tpc + self.lmbda * nn.ReLU()(g) + self.lmbda2 * nn.ReLU()(-tpc - fpc)

        tpc = torch.dot(y.flatten(), 1 - self.leaky_relu_01_loss(f, 1.0, self.leaky_slope).flatten())
        fpc = torch.dot(y.flatten(), self.leaky_relu_01_loss(f, 1.0, self.leaky_slope).flatten())
        fnc = torch.dot(1 - y.flatten(), self.leaky_relu_01_loss(f, 0.0, self.leaky_slope).flatten())

        # loss = (
        #     (1 + self.lmbda) * fpc
        #     + self.lmbda * self.min_prec / (1.0 - self.min_prec) * fnc
        #     - self.lmbda * Nplus
        #     # + self.lmbda2 * (-tpc - fpc)
        # )

        # g = -tpc + self.min_prec / (1.0 - self.min_prec) * fpc + Nplus

        # loss = -tpc + self.lmbda * g + self.lmbda2 * (-tpc - fpc)

        g = fpc + self.min_prec / (1.0 - self.min_prec) * fnc - Nplus

        loss = fpc + self.lmbda * nn.ReLU()(g)  # + self.lmbda2 * nn.ReLU()(-tpc - fpc)

        return loss
