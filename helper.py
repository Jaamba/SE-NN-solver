import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch


# Generates a batch of random torch tensors representing continous functions.
# n is the number of points of the function (discretization size).
def random_function(batch_size, n, sigma=8, device="cuda"):

    # Generates white noise
    x = torch.randn(batch_size, 1, n, device=device)

    # Smooths out the white noise using a moving average
    k = int(6 * sigma + 1)
    if k % 2 == 0:
        k += 1

    t = torch.arange(k, device=device) - k // 2

    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel /= kernel.sum()

    y = F.conv1d(
        F.pad(x, (k // 2, k // 2), mode="reflect"),
        kernel.view(1, 1, -1)
    ).squeeze(1)

    # Window equal to zero at the boundaries
    window = torch.hann_window(n, periodic=False, device=device)

    return y * window

# Does the second derivative of a batch of functions 
def second_derivative(f, dt):
    # f: (batch_size, N)

    kernel = torch.tensor([1., -2., 1.],
                          device=f.device,
                          dtype=f.dtype).view(1, 1, 3) / dt**2

    # Does the derivative. Cuda compatible
    d2 = F.conv1d(f.unsqueeze(1), kernel).squeeze(1)

    out = torch.empty_like(f)

    # Inner points
    out[:, 1:-1] = d2

    # Boundaries
    out[:, 0] = (2*f[:,0] - 5*f[:,1] + 4*f[:,2] - f[:,3]) / dt**2
    out[:, -1] = (2*f[:,-1] - 5*f[:,-2] + 4*f[:,-3] - f[:,-4]) / dt**2

    return out

# Applies the hamiltonian operator to a batch of functions
# Note that this is whithout constant, so the whole thing is adimensional
def hamiltonian(f, V, dt):
    return V*f - second_derivative(f, dt) 
