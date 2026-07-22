import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import helper

# The set on which the SE will be solved will be like [-A, A]
A = 10
# Discretization size of the set
N = 512
t = np.linspace(-A, A, N)
dt = t[1] - t[0]

# Network
class SESolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, N)

        # Mask needed to impose phi = 0 at boundaries
        x = torch.linspace(-A, A, N)
        mask = 1 - (x / A)**2
        self.register_buffer("boundary_mask", mask)

    def forward(self, x):
        V = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        phi = x

        # Imposes phi = 0 at boundaries
        phi = phi * self.boundary_mask

        # Makes sure that the output function is normalized
        integral = torch.sqrt(dt * torch.sum(phi**2, dim=1, keepdim=True))
        phi = phi / integral

        # Calculates E based on phi using E = <phi|H|phi>
        Hphi = helper.hamiltonian(phi, V, dt)
        E = dt * torch.sum(phi * Hphi, dim=1, keepdim=True)

        return E, phi, Hphi