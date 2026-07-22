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

# Solves the schrodinger equation using finite differences method
def solve_schrodinger(V, dt):
    """
    Solves

        (-d²/dx² + V) phi = E phi

    using finite differences and diagonalization.

    Parameters
    ----------
    V : torch.Tensor
        Potential of shape (N,) on CPU or CUDA.
    dt : float
        Spatial step.

    Returns
    -------
    E : torch.Tensor
        Eigenvalues, shape (N,)
    phi : torch.Tensor
        Eigenvectors, shape (N, N).
        Column i is the eigenfunction corresponding to E[i].
    """

    device = V.device
    dtype = V.dtype
    N = V.numel()

    inv_dt2 = 1.0 / (dt * dt)

    # Main diagonal
    diag = 2.0 * inv_dt2 + V

    # Off-diagonal
    off = -inv_dt2 * torch.ones(N - 1, device=device, dtype=dtype)

    # Hamiltonian
    H = torch.diag(diag)
    H += torch.diag(off, diagonal=1)
    H += torch.diag(off, diagonal=-1)

    # Solve H phi = E phi
    E, phi = torch.linalg.eigh(H)

    # Normalize eigenfunctions so that
    # sum |phi|² dx = 1
    norm = torch.sqrt(torch.sum(phi**2, dim=0) * dt)
    phi = phi / norm

    return E, phi

N = 1000
t = np.linspace(-10, 10, N)
dt = t[1] - t[0]
V = torch.empty(N)
V[:] = 0
V[int(40*N/100):int(60*N/100)] = -1

E, phi = solve_schrodinger(V, dt)
groundState = phi[:, 0].squeeze()
firstExcited = phi[:, 1].squeeze()
plt.plot(t, groundState.detach().cpu().numpy())
plt.plot(t, firstExcited.detach().cpu().numpy())
plt.plot(t, phi[:, 4].detach().cpu().numpy())
plt.plot(t, V.detach().cpu().numpy())
plt.show()
print(E[2])
