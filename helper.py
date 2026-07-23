import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch


# Generates a batch of random torch tensors representing continous functions.
# n is the number of points of the function (discretization size).
def random_smooth(batch_size, n, sigma=8, device="cuda"):

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
    y *= window

    # normalizes the functions
    y /= y.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)

    return y

# Generates a batch of M random polynomials of degree with N points using cuda
def random_polynomials(M, N, degree=8, device="cuda"):
    x = torch.linspace(-1, 1, N, device=device)

    # (M, degree+1)
    coeffs = torch.randn(M, degree + 1, device=device)

    # (degree+1, N)
    powers = torch.stack([x**k for k in range(degree + 1)])

    # (M, N)
    f = coeffs @ powers

    # Makes sure that f = 0 at boundaries
    mask = 1 - x**8
    f *= mask

    # normalizes the functions
    f /= f.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)

    return f

# Generates a batch of randomly placed and deep gausian wells using cuda
def random_gaussian_wells( M, N, max_wells=4, well_steep = 2, device="cuda"):

    # Assumes the function is between -1 and 1 (can be stretched)
    x = torch.linspace(-1, 1, N, device=device)
    x = x[None, None, :]              # (1,1,N)

    K = max_wells
    well_steep = 2*well_steep # well steep has to be even

    # Choses the positions of the centers of the well between -1 and 1
    centers = torch.rand(M, K, device=device) * 2 - 1

    # Generates random widths between 0.05 and 0.30
    widths = 0.05 + 0.25 * torch.rand(M, K, device=device)
    # Generates random depths between -1 and +1
    depths = 2 * torch.rand(M, K, device=device) - 1

    centers = centers[:, :, None]
    widths = widths[:, :, None]
    depths = depths[:, :, None]

    # Creates the wells
    wells = depths * torch.exp(
        -(x - centers)**well_steep / (2 * widths**well_steep)
    )

    # Sums them
    V = wells.sum(dim=1)

    # Makes sure that the function is 0 at boundaries
    mask = 1 - torch.linspace(-1,1,N,device=device)**8
    V *= mask

    return V


### Testing of random function generators
t = np.linspace(-10, 10, 512)

f1 = random_polynomials(1, 512, device="cpu")
f1np = f1.squeeze().detach().numpy()
plt.plot(t, f1np, label = "polynomial")

f2 = random_gaussian_wells(1, 512, device="cpu", max_wells=3, well_steep=4)
f2np = f2.squeeze().detach().numpy()
plt.plot(t, f2np, label = "well")

f4 = random_smooth(1, 512, device="cpu", sigma=36)
f4np = f4.squeeze().detach().numpy()
plt.plot(t, f4np, label="smooth")

plt.legend()
plt.show()

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

def solve_schrodinger(V, dt, n):
    """
    Solves

        (-d²/dx² + V) phi = E phi

    using finite differences.

    Parameters
    ----------
    V : torch.Tensor
        Shape (M, N) representing batch of potentials.

    dt : float
        Spatial step.

    n : integer
        Energy level

    Returns
        E   : (M, 1)
        phi : (M, N)
    """

    device = V.device
    dtype = V.dtype

    inv_dt2 = 1.0 / (dt * dt)

    # Checks that V has shape (M, N)
    if V.ndim == 2:
        M, N = V.shape

        # Creates the batched hamiltonian matrix
        diag = 2.0 * inv_dt2 + V
        off = -inv_dt2 * torch.ones(N - 1, device=device, dtype=dtype)

        H = torch.diag_embed(diag)

        idx = torch.arange(N - 1, device=device)
        H[:, idx, idx + 1] = off
        H[:, idx + 1, idx] = off

        # Diagonalization
        E, phi = torch.linalg.eigh(H)

        # Returns the n-th state
        E = E[:, n:n+1]
        phi = phi[:, :, n]

        # Normalizes the result
        norm = torch.sqrt(torch.sum(phi**2, dim=1, keepdim=True) * dt)
        phi = phi / norm

        return E, phi

    else:
        raise ValueError("V must have shape (N,) or (M, N)")

# N = 1000
# t = np.linspace(-10, 10, N)
# dt = t[1] - t[0]
# V = torch.empty(N)
# V[:] = 0
# V[int(40*N/100):int(60*N/100)] = -1
# V = V.unsqueeze(0)
# V = random_function(2, N, device="cpu")

# E0, phi0 = solve_schrodinger(V, dt, 0)
# E1, phi1 = solve_schrodinger(V, dt, 1)
# plt.plot(t, phi0[0].squeeze().detach().cpu().numpy())
# plt.plot(t, phi0[1].squeeze().detach().cpu().numpy())
# plt.plot(t, V[0].squeeze().detach().cpu().numpy())
# plt.plot(t, V[1].squeeze().detach().cpu().numpy())
# plt.show()
# print("Ground state = " + str(E0.squeeze().item()))
# print("First excited = " + str(E1.squeeze().item()))
