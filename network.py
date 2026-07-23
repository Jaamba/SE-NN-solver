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

        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.ReLU(),

            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.ReLU(),

            nn.Conv1d(64, 1, kernel_size=9, padding=4)
        )

        # Mask needed to impose phi = 0 at boundaries
        x = torch.linspace(-A, A, N)
        mask = 1 - (x / A)**2
        self.register_buffer("boundary_mask", mask)

    def forward(self, V):

        x = V.unsqueeze(1)
        phi = self.conv(x)
        phi = phi.squeeze(1)

        # Imposes phi = 0 at boundaries
        phi = phi * self.boundary_mask

        # Makes sure that the output function is normalized
        integral = torch.sqrt(dt * torch.sum(phi**2, dim=1, keepdim=True))
        phi = phi / integral

        # Calculates E based on phi using E = <phi|H|phi>
        Hphi = helper.hamiltonian(phi, V, dt)
        E = dt * torch.sum(phi * Hphi, dim=1, keepdim=True)

        return E, phi, Hphi