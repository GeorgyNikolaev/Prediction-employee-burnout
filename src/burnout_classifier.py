import torch
import torch.nn as nn


class BurnoutClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(BurnoutClassifier, self).__init__()
        self.fcs = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 1)

        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fcs(x).squeeze(1)
        out = self.sigmoid(out)
        return out

    def predict(self, x, threshold=0.5):
        out = self.forward(x)
        return (out > threshold).int()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
