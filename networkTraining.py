import torch
from torch import optim
import network
import helper
import datasetGen
import torch.nn.functional as F

# Uses the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: " + str(device))

# Build the network
model = network.SESolver().to(device)

# Loss function definition for the network
def criterion(E, phi, Hphi, V, Ebatch):
    # Minimizing this will make sure that phi is a solution. Only considers inner values
    mse = torch.mean(
        (Hphi[:,1:-1] - Ebatch*phi[:,1:-1])**2
    )

    # Rewards the solver if it selected energies close to the ground state
    groundLoss = torch.mean( (E - Ebatch)**2 )

    return mse + groundLoss

# Model optimizer to train the model
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Number of epochs
epoch = 1

# Loads the dataset
print("Loading dataset...")
dataset = torch.load( "trainingset.pt", map_location="cpu")
print("Dataset loaded corretctly")

# Loads the Energy set
print("Loading energy set...")
Eset = torch.load( "ground_energy_set.pt", map_location="cpu")

BATCH_SIZE = datasetGen.BATCH_SIZE
NUM_BATCHES = datasetGen.NUM_BATCHES

# Makes sure to be in training mode
model.train()
for i, (batch, Ebatch) in enumerate(zip(dataset, Eset)):

    # Moves the batch to the used device
    batch = batch.to(device, non_blocking=True)
    Ebatch = Ebatch.to(device, non_blocking=True)

    # Prints current training progress
    if i % (NUM_BATCHES/100) == 0:
        print(f"Current progress: {i/NUM_BATCHES*100}%", end="\r") 

    for j in range(epoch):
        optimizer.zero_grad()

        # Forward press
        E, phi, Hphi = model(batch)

        # Calculates the loss
        loss = criterion(E, phi, Hphi, batch, Ebatch)

        # Bacward press
        loss.backward()
        optimizer.step()
print()

# Saves the trained model
torch.save(model.state_dict(), 'checkpoint.pth')