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

# Asks to load previously trained network  if available
if(input("Do you want to load previously trained network? (y/N)").lower() == "y"):
    state_dict = torch.load('checkpoint.pth')
    model.load_state_dict(state_dict)

# Loss function definition for the network
def criterion(E):
    return torch.mean(E)

# Model optimizer to train the model
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Number of epochs
epoch = 1

# Loads the dataset
print("Loading dataset...")
dataset = torch.load( "trainingset.pt", map_location="cpu")
print("Dataset loaded corretctly")

# Loads the Energy set
# print("Loading energy set...")
# Eset = torch.load( "ground_energy_set.pt", map_location="cpu")
# print("Energy set loaded correctly")

BATCH_SIZE = datasetGen.BATCH_SIZE
NUM_BATCHES = datasetGen.NUM_BATCHES

# Makes sure to be in training mode
model.train()
for i, batch in enumerate(dataset):

    # Moves the batches to the used device
    batch = batch.to(device, non_blocking=True)

    # Prints current training progress
    print(f"\rTraining network: {i+1}/{NUM_BATCHES} ({100*i/NUM_BATCHES:.1f}%)", end="", flush=True)

    for j in range(epoch):
        optimizer.zero_grad()

        # Forward press
        E, phi, Hphi = model(batch)

        # Calculates the loss
        loss = criterion(E)

        # Bacward press
        loss.backward()
        optimizer.step()
print()

# Saves the trained model
torch.save(model.state_dict(), 'checkpoint.pth')