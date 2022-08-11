from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from constrained_metric_loss.sigmoid_optimizer import tight_sigmoid


def get_sigmoid_params(gamma, delta, eps, upper_or_lower, epochs=1000):
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

    def extract_sigmoid_hyperparameters(self, sigmoid_hyperparams):
        self.gamma = sigmoid_hyperparams["gamma"]
        self.delta = sigmoid_hyperparams["delta"]
        self.eps = sigmoid_hyperparams.get("eps", None)

    def extract_or_calculate_sigmoid_params(self, sigmoid_params):

        if sigmoid_params is None:
            self.mtilde, self.btilde = get_sigmoid_params(self.gamma, self.delta, self.eps, "lower")
            self.mhat, self.bhat = get_sigmoid_params(self.gamma, self.delta, self.eps, "upper")
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
        loss = -tpc + self.lmbda * nn.ReLU()(g)
        # The reason for the odd way of calling the ReLU function:
        # https://discuss.pytorch.org/t/multiplication-of-activation-function-with-learnable-parameter-scalar/113746/2

        return loss


# def min_prec_loss(
#     f,
#     y: np.arrray,
#     min_prec: float,
#     lmbda: float,
#     sigmoid_hyperparams: Optional[Dict] = None,
#     sigmoid_params: Optional[Dict] = None,
# ):

#     gamma = sigmoid_hyperparams["gamma"]
#     delta = sigmoid_hyperparams["delta"]
#     eps = sigmoid_hyperparams.get("eps", None)

#     if sigmoid_params is None:
#         mtilde, btilde = get_sigmoid_params(gamma, delta, eps, "lower")
#         mhat, bhat = get_sigmoid_params(gamma, delta, eps, "upper")
#     else:
#         mtilde = sigmoid_params["mtilde"]
#         btilde = sigmoid_params["btilde"]
#         mhat = sigmoid_params["mhat"]
#         bhat = sigmoid_params["bhat"]

#     # Eqn. 14
#     tpc = torch.sum(
#         torch.where(
#             y == 1.0,
#             (1 + gamma * delta) * torch.sigmoid(mtilde * f + btilde),
#             torch.tensor(0.0),
#         )
#     )

#     # Eqn 10
#     fpc = torch.sum(torch.where(y == 0.0, (1 + gamma * delta) * torch.sigmoid(mhat * f + bhat), torch.tensor(0.0)))

#     # Line below eqn. 1 in paper
#     Nplus = torch.sum(y)

#     # Eqn. 12
#     g = -tpc + min_prec / (1.0 - min_prec) * fpc + gamma * delta * Nplus

#     # Eqn. 12
#     loss = -tpc + lmbda * nn.ReLU()(g)
#     # The reason for the odd way of calling the ReLU function:
#     # https://discuss.pytorch.org/t/multiplication-of-activation-function-with-learnable-parameter-scalar/113746/2

#     return loss
