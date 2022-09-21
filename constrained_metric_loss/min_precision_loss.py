from typing import Dict, Optional


import torch
import torch.nn as nn

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
        tpc = torch.sum(
            torch.where(
                y == 1.0,
                (1 + self.gamma * self.delta) * torch.sigmoid(self.mtilde * f + self.btilde),
                torch.tensor(0.0),
            )
        )

        # Eqn 10
        fpc = torch.sum(
            torch.where(
                y == 0.0, (1 + self.gamma * self.delta) * torch.sigmoid(self.mhat * f + self.bhat), torch.tensor(0.0)
            )
        )

        # Line below eqn. 1 in paper
        Nplus = torch.sum(y)

        # Eqn. 12
        g = -tpc + self.min_prec / (1.0 - self.min_prec) * fpc + self.gamma * self.delta * Nplus

        # Eqn. 12
        loss = -tpc + self.lmbda * torch.pow(nn.ReLU()(g), 2) #+ Nplus*fpc
        # The reason for the odd way of calling the ReLU function:
        # https://discuss.pytorch.org/t/multiplication-of-activation-function-with-learnable-parameter-scalar/113746/2

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
