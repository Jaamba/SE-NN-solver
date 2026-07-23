import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch
import network

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

# Generates and saves a dataset of functions for training. This is a tensor of
# shape (num_batches, batch_size, N)
def generate_training_set( filename, device="cuda"):

    # Generates an empy dataset
    dataset = torch.empty(
        NUM_BATCHES,
        BATCH_SIZE,
        N,
        dtype=torch.float32
    )

    for b in range(NUM_BATCHES):

        # Generates three batches of different type
        poly   = random_polynomials(BATCH_SIZE, N, device=device)
        smooth = random_smooth(BATCH_SIZE, N, device=device)
        wells  = random_gaussian_wells(BATCH_SIZE, N, device=device)

        # Generates a batch three random coefficients between 0 and 1
        weights = torch.rand(BATCH_SIZE, 3, device=device)

        # Decides if a generator is used
        active = (torch.rand(BATCH_SIZE, 3, device=device) < 0.7).float()

        # Avoids all being 0
        inactive = active.sum(dim=1) == 0 # boolean mask to find inactive spots
        active[inactive, torch.randint(0, 3, (inactive.sum(),), device=device)] = 1

        # Creates the batch
        weights *= active
        batch = (
            weights[:, 0:1] * poly +
            weights[:, 1:2] * smooth +
            weights[:, 2:3] * wells
        )

        # Updates the dataset with the current batch
        dataset[b] = batch.cpu()

        # Prints current training progress
        if b % (NUM_BATCHES/100) == 0:
            print(f"Current progress: {b/NUM_BATCHES*100}%", end="\r") 

    torch.save(dataset, filename)

    print(f"\nDataset saved in '{filename}'")

### DATASET INFO
BATCH_SIZE = 512
NUM_BATCHES = 10000
N = network.N

# Generates the training set
generate_training_set('trainingset.pt', device="cuda")
