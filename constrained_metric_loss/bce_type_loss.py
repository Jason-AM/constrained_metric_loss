import torch
import torch.nn as nn


class BCELogitManual(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f, y):
        def _logsumexp(f):
            return torch.logsumexp(torch.column_stack((torch.zeros(len(f), 1), (-f).reshape(-1, 1))), 1)

        # tpc = torch.sum(torch.where(y == 1.0, torch.log(torch.tensor(1)) - _logsumexp(f), torch.tensor(0.0)))

        # tnc = torch.sum(torch.where(y == 0.0, torch.log(torch.tensor(1)) - _logsumexp(1 - f), torch.tensor(0.0)))

        tpc = torch.sum(torch.where(y == 1.0, torch.log(torch.sigmoid(f)), torch.tensor(0.0)))

        tnc = torch.sum(torch.where(y == 0.0, torch.log(torch.sigmoid(1 - f)), torch.tensor(0.0)))

        return -tpc - tnc
