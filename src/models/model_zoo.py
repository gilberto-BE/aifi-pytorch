import torch
import torch.nn as nn
import torch.functional as F
from math import copy


def clone_layers(module, N):
    """Produce N identical layers.
    
    Args:
    * module: A pytorch layer class or other object considered a pytorch module.
    * N: Number of copies of the module passed to the function.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class NBeatsBlock:
    def __init__(self, in_features, out_features, n_layers=4):
        super(NBeatsBlock, self).__init__()
        self.in_features = in_features 
        self.out_features = out_features
        self.n_layers = n_layers
        self.fc_layers = clone_layers(
            nn.Linear(self.in_features, self.in_features), 
            self.n_layers
            )
        self.bc_layer = nn.Linear(self.in_features, self.in_features)
        self.backcast = nn.Linear(self.in_features, self.out_features)
        self.forecast = nn.Linear(self.in_features, self.out_features)

    def forward(self, x):
        residual = x
        for layer in self.fc_layers:
            x = F.relu(layer(x))
        backcast = F.relu(self.bc_layer(residual - x))
        return  self.backcast(backcast), self.forecast(x)
