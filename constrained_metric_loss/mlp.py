import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-Layer Perceptron"""

    def __init__(self, nfeat, hidden_width, loss, loss_arguments):
        super().__init__()

        self.nfeat = nfeat

        self.hidden_width = hidden_width
        self.loss = loss
        self.loss_arguments = loss_arguments

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.nfeat, self.hidden_width),
            nn.ReLU(),
            nn.Linear(self.hidden_width, 1),
        )

        self.loss_func = self.loss(**self.loss_arguments)

    def forward(self, x, y):
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y)

        self.f = self.layers(x)

        return self.loss_func(self.f, y.float())

    def sigmoid_of_score(self, xtest):

        xtest = torch.from_numpy(xtest).float()

        f = self.layers(xtest)
        return torch.sigmoid(f).detach().numpy().flatten()

    def decision_score(self, xtest):

        xtest = torch.from_numpy(xtest).float()

        f = self.layers(xtest)
        return f.detach().numpy().flatten()

    def fit(self, x, y, optimizer, n_batches=200):

        training_loss = []
        for _ in range(n_batches):
            optimizer.zero_grad()
            loss = self.forward(x, y)
            loss.backward()
            optimizer.step()

            training_loss.append(loss.detach().numpy())
        return training_loss

    def fit_w_lambda_grad_ascent(self, x, y, optimizer, optimizer_lam, n_batches=200):
        training_loss = []
        for _ in range(n_batches):

            optimizer.zero_grad()
            model_params = list(self.parameters())[0]
            loss = self.forward(x, y)
            loss.backward(inputs=model_params)
            optimizer.step()

            optimizer_lam.zero_grad()
            lam = list(self.parameters())[1:]
            loss_lam = -self.forward(x, y)
            loss_lam.backward(inputs=lam)
            optimizer_lam.step()

            training_loss.append(loss.detach().numpy())

        return training_loss
