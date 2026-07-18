import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import torch.functional as F
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
    )

    # Squeezes the batch
    return y.squeeze(1)



