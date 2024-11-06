import torch.nn as nn


class BurnoutClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(BurnoutClassifier, self).__init__()
        self.fcs = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.fcs(x)
        return out

    def predict(self, x, threshold=0.5):
        out = self.forward(x)
        return (out > threshold).int()
