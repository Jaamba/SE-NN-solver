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
            nn.Conv1d(2, 64, kernel_size=9, padding=4),
            nn.ReLU(),

            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.ReLU(),

            nn.Conv1d(128, 64, kernel_size=9, padding=4),
            nn.ReLU(),

            nn.Conv1d(64, 1, kernel_size=9, padding=4)
        )

        # Mask needed to impose phi = 0 at boundaries
        mask = torch.ones(N)
        mask[0]=0
        mask[-1]=0
        self.register_buffer("boundary_mask", mask)

    def forward(self, V):

        # Positional chanel for the network
        pos = torch.linspace(-A,A,N,device=V.device)
        pos = pos.expand(V.shape[0],1,N)

        x = torch.cat(
            [V.unsqueeze(1), pos],
            dim=1
        )

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

        return E, phi

# Different architecture intended for operating on fourier decomposition
# of the input
class FourierNet(nn.Module):
    def __init__(self, modes=128):
        super().__init__()

        # Modes to do the decomposition on
        self.modes = modes

        # Operates on both immaginary and real frequencies
        self.net = nn.Sequential(
            nn.Linear(2*modes, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2*modes)
        )

        # Mask needed to impose phi = 0 at boundaries
        mask = torch.ones(N)
        mask[0]=0
        mask[-1]=0
        self.register_buffer("boundary_mask", mask)

    def forward(self, x):

        # Real FFT of input
        F = torch.fft.rfft(x)

        # takes the first modes
        low = F[:, :self.modes]

        # Divides comples and real values
        low_real = torch.view_as_real(low)      # (batch,M,2)
        low_real = low_real.flatten(1)          # (batch,2M)

        # Neural network
        out = self.net(low_real)

        # Goes back to comples mode
        out = out.view(-1, self.modes, 2)
        out = torch.view_as_complex(out)

        # Same shape as F needed to do inverse FFT
        F_new = torch.zeros_like(F)
        F_new[:, :self.modes] = out

        # inverse FFT
        phi = torch.fft.irfft(F_new, n=N)

        # Imposes phi = 0 at boundaries
        phi = phi * self.boundary_mask

        # Makes sure that the output function is normalized
        integral = torch.sqrt(dt * torch.sum(phi**2, dim=1, keepdim=True))
        phi = phi / integral

        # Calculates E based on phi using E = <phi|H|phi>
        Hphi = helper.hamiltonian(phi, x, dt)
        E = dt * torch.sum(phi * Hphi, dim=1, keepdim=True)

        return E, phi