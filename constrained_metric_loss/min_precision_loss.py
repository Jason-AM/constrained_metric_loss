import torch
import torch.nn as nn
import torch.optim as optim

from constrained_metric_loss.sigmoid_optimizer import tight_sigmoid


def get_sigmoid_params(gamma, delta, eps, upper_or_lower, epochs=1000):
    model = tight_sigmoid(gamma, delta, eps, upper_or_lower)
    optimizer = optim.Adam(model.parameters(), lr=1)

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()

    m, b = model.named_parameters()
    m = float(m[1].detach().numpy())
    b = float(b[1].detach().numpy())
    return m, b


def min_prec_loss(f, y, min_prec, lmbda, sigmoid_hyperparams):
    gamma, delta, _ = sigmoid_hyperparams

    mtilde, btilde = get_sigmoid_params(*sigmoid_hyperparams, "lower")
    mhat, bhat = get_sigmoid_params(*sigmoid_hyperparams, "upper")

    print(mtilde, btilde, mhat, bhat)

    # Eqn. 14
    tpc = torch.sum(
        torch.where(
            y == 1.0,
            (1 + gamma * delta) * torch.sigmoid(mtilde * f + btilde),
            torch.tensor(0.0),
        )
    )

    # Eqn 10
    fpc = torch.sum(torch.where(y == 0.0, (1 + gamma * delta) * torch.sigmoid(mhat * f + bhat), torch.tensor(0.0)))

    # Line below eqn. 1 in paper
    Nplus = torch.sum(y)

    # Eqn. 12
    g = -tpc + min_prec / (1.0 - min_prec) * fpc + gamma * delta * Nplus

    # Eqn. 12
    loss = -tpc + lmbda * nn.ReLU()(g)
    # The reason for the odd way of calling the ReLU function:
    # https://discuss.pytorch.org/t/multiplication-of-activation-function-with-learnable-parameter-scalar/113746/2

    return loss
