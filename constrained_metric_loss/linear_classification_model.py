import numpy as np
import torch
import torch.nn as nn


class LinearClassificationModel(nn.Module):
    """Produces classification scores for given loss"""

    def __init__(
        self,
        nfeat,
        model_param_init,
        loss,
        loss_arguments,
    ):
        super().__init__()

        self.nfeat = nfeat
        self.beta = nn.Parameter(torch.from_numpy(model_param_init).float())
        assert nfeat + 1 == len(model_param_init), "Using the coefficient trick so need to add one extra parameter"

        self.loss = loss
        self.loss_arguments = loss_arguments

        self.loss_func = self.loss(**self.loss_arguments)

    def forward(self, x, y):
        x = np.column_stack((x, np.ones([x.shape[0]])))
        x = torch.from_numpy(x).float()

        self.f = x @ self.beta

        return self.loss_func(self.f, y.float())

    def predict_proba(self, xtest):

        if len(xtest.shape) == 1:
            xtest = xtest.reshape([1, self.nfeat])

        xtest = np.column_stack((xtest, np.ones([xtest.shape[0]])))
        xtest = torch.from_numpy(xtest).float()

        f = xtest @ self.beta
        return torch.sigmoid(f).detach().numpy().flatten()

    def fit(self, x, y, optimizer, n_batches=200):
        training_loss = []
        for i in range(n_batches):
            optimizer.zero_grad()
            loss = self.forward(x, y)
            loss.backward()
            optimizer.step()

            training_loss.append(loss.detach().numpy())

        return training_loss
