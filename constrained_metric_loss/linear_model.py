import numpy as np
import torch
import torch.nn as nn


class LinearModel(nn.Module):
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
        y = torch.from_numpy(y)

        self.f = x @ self.beta

        return self.loss_func(self.f, y.float())

    def sigmoid_of_score(self, xtest):

        if len(xtest.shape) == 1:
            xtest = xtest.reshape([1, self.nfeat])

        xtest = np.column_stack((xtest, np.ones([xtest.shape[0]])))
        xtest = torch.from_numpy(xtest).float()

        f = xtest @ self.beta
        return torch.sigmoid(f).detach().numpy().flatten()

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


class InitialParameterGenerator:
    def __init__(self, seed: np.array):
        # defines a starting set of initial conditions - needed even if just for th shape
        self.seed = seed
        self.initial_parameters = [seed]
        self.rng = np.random.default_rng(0)

    @staticmethod
    def generate_full_grid(param_seed, number_per_dim, seed_multiplier=10):

        linspaces = [
            np.linspace(-seed_multiplier * seed, seed_multiplier * seed, number_per_dim) for seed in param_seed
        ]
        n_times_away_grid = np.meshgrid(*linspaces)
        search_space = [dim_params.flatten() for dim_params in n_times_away_grid]

        return np.column_stack(search_space)

    def generate_samples_from_grid(self, num_to_select=10, number_per_dim=2, seed_multiplier=10):

        random_params_centered_on_seed = self.rng.choice(
            self.generate_full_grid(self.seed, number_per_dim=number_per_dim, seed_multiplier=seed_multiplier),
            size=num_to_select,
        )
        self.initial_parameters = np.vstack([self.initial_parameters, random_params_centered_on_seed])

    def generate_normally_distributed_around_seed(self, number=6, var=1, shift=0):
        random_params_from_normal_centered_at_seed = self.seed + self.rng.normal(
            shift, var, size=(number, len(self.seed))
        )
        self.initial_parameters = np.vstack([self.initial_parameters, random_params_from_normal_centered_at_seed])

    def generate_normally_distributed(self, number=6, mean=0, var=1):
        random_params_from_normal = self.rng.normal(mean, var, size=(number, len(self.seed)))
        self.initial_parameters = np.vstack([self.initial_parameters, random_params_from_normal])
