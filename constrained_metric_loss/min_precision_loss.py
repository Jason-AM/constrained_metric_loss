from typing import Dict, Optional
import inspect

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
        y = y.flatten()
        f = f.flatten()

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
        g = -tpc + (self.min_prec / (1.0 - self.min_prec)) * fpc + self.gamma * self.delta * Nplus

        # Eqn. 12
        loss = -tpc + self.lmbda * nn.ReLU()(g)

        return loss


class ZeroOneApproximation:
    def __init__(self, name_of_approx: str, params_dict: dict):

        # ensure name chosen from available options
        try:
            available_approxs = [name for name in dir(self) if "__" not in name]
            assert name_of_approx in available_approxs
            self.zero_one_loss_approx_func = getattr(self, name_of_approx)
        except AssertionError:
            raise ValueError(
                "chosen approximation not in list of possible options. "
                "Please select one of the following: "
                f"{available_approxs}"
            )

        # ensure the params dictionary contains the correct inputs for the chosen name
        try:
            required_params = inspect.getfullargspec(self.zero_one_loss_approx_func).args
            assert all(item in params_dict.keys() for item in required_params)
            self.zero_one_loss_approx_func = self.zero_one_loss_approx_func(**params_dict)
        except AssertionError:
            raise ValueError(
                "Parameters dictionary is missing arguments for chosen loss approximation. "
                f"For your chosen loss {name_of_approx} you need to define: "
                f"{required_params}"
            )

    @staticmethod
    def leaky_relu_approx(negative_slope):
        return lambda x, y: nn.ReLU()(1 - nn.LeakyReLU(negative_slope=negative_slope)((2 * y - 1) * x))

    @staticmethod
    def logsigmoid_approx():
        return lambda x, y: -torch.log(torch.sigmoid((2 * y - 1) * x))

    @staticmethod
    def semi_rath_hughes_approx(gamma, delta, eps):
        mhat, bhat = get_sigmoid_params_bfgs(gamma, delta, eps, "upper")
        return lambda x, y: (1 + gamma * delta) * torch.sigmoid(-(2 * y - 1) * mhat * x + bhat)

    def calc_loss(self, x, y):
        return self.zero_one_loss_approx_func(x, y)


class MinPrec01LossApprox(nn.Module):
    def __init__(self, min_prec: float, lmbda: float, lmbda2: float, zero_one_loss_approx: ZeroOneApproximation):
        super().__init__()
        self.min_prec = min_prec
        self.lmbda = lmbda
        self.lmbda2 = lmbda2
        self.zero_one_loss_approx = zero_one_loss_approx

    def forward(self, f, y):

        tpc = torch.dot(y.flatten(), 1 - self.zero_one_loss_approx.calc_loss(f, 1.0).flatten())
        fpc = torch.dot(1 - y.flatten(), self.zero_one_loss_approx.calc_loss(f, 0.0).flatten())

        Nplus = torch.sum(y)

        g = -tpc + self.min_prec / (1.0 - self.min_prec) * fpc + Nplus

        loss = -tpc + self.lmbda * nn.ReLU()(g) + self.lmbda2 * nn.ReLU()(-tpc - fpc)

        return loss
