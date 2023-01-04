import math
import torch
from torch import nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self, dim):
        super(Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 1500, bias=True),
            nn.ReLU(),
            nn.Linear(1500, 500, bias=True),
            nn.ReLU(),
            nn.Linear(500, 50, bias=True),
            nn.ReLU(),
            nn.Linear(50, 10, bias=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 50, bias=True),
            nn.ReLU(),
            nn.Linear(50, 500, bias=True),
            nn.ReLU(),
            nn.Linear(500, 1500, bias=True),
            nn.ReLU(),
            nn.Linear(1500, dim, bias=True),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(10, 5, bias=True),
            nn.ReLU(),
            nn.Linear(5, 5, bias=True),
            nn.ReLU(),
            nn.Linear(5, 1, bias=True),
        )

    def encode(self, x):
        x = self.encoder(x)
        x = F.normalize(x)
        return x

    def decode(self, x):
        return self.decoder(x)

    def clasy(self, x):
        return self.classifier(x)

    def predict(self, x):
        z_new = self.encode(x)
        prediction = self.clasy(z_new)
        return prediction
