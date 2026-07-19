import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# The set on which the SE will be solved will be like [-A, A]
A = 30
# Discretization size of the set
N = 1024
t = np.linspace(-A, A, N)
dt = t[1] - t[0]

# Network
class SESolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, N+1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        E = x[:, 0]
        E = E.unsqueeze(1)
        phi = x[:, 1:]

        # Makes sure that the output function is normalized
        integral = torch.sqrt(dt * torch.sum(phi**2, dim=1, keepdim=True))
        phi = phi / integral

        return E, phi