import torch
from torch import optim
from torch import nn
import numpy as np
import network
import helper
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Uses the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Build the network
model = network.SESolver().to(device)

# Loss function definition for the network
def criterion(E, phi, V):
    return torch.mean( (helper.hamiltonian(phi, V, network.dt) - E*phi)**2 )

# Model optimizer to train the model
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Number of batches to generate
pols = 50000
# batch size
BATCH_SIZE = 1024
# Number of epochs
epoch = 3

# Makes sure to be in training mode
model.train()
for i in range(pols):

    # Prints current training progress
    if i % (pols/100) == 0:
        print("Current progress:", i/pols*100, "%") 

    # Input values for the network
    input = helper.random_function(BATCH_SIZE, network.N, device=device)

    for j in range(epoch):
        optimizer.zero_grad()

        # Forward press
        E, phi = model(input)

        # Calculates the loss
        loss = criterion(E, phi, input)

        # Bacward press
        loss.backward()
        optimizer.step()

# Saves the trained model
torch.save(model.state_dict(), 'checkpoint.pth')