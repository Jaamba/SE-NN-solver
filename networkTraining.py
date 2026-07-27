import torch
from torch import optim
import network
import helper
import datasetGen
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Uses the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: " + str(device))

# Build the network
model = network.FourierNet().to(device)

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
epoch = 15

# Loads the dataset
print("Loading dataset...")
dataset = torch.load( "trainingset.pt", map_location="cpu")
print("Dataset loaded corretctly")

# Imports information from datasetGen
BATCH_SIZE = datasetGen.BATCH_SIZE
NUM_BATCHES = datasetGen.NUM_BATCHES

# Losses for each epoch
train_losses = []

# Makes sure to be in training mode
model.train()
for j in range(epoch):

    # Shuffles the dataset
    datasetGen.shuffleDataset(dataset)
    running_loss = 0

    for i, batch in enumerate(dataset):
        optimizer.zero_grad()

        # Moves the batches to the used device
        batch = batch.to(device, non_blocking=True)

        # Forward press
        E, phi = model(batch)

        # Calculates the loss
        loss = criterion(E)
        running_loss += loss.item()

        # Bacward press
        loss.backward()
        optimizer.step()

        # Prints current progress
        print(
            f"\rTraining network: Epoch {j+1}/{epoch}; "
            f"Batch {i+1}/{NUM_BATCHES}; "
            f"Total progress: {100*(j*NUM_BATCHES + i + 1)/(NUM_BATCHES*epoch):.1f}%",
            end="",
            flush=True,
        )
    else:
        train_losses.append(running_loss/NUM_BATCHES)
print()

# Saves the trained model
torch.save(model.state_dict(), 'checkpoint.pth')

# Prints information about the training progress
plt.plot(train_losses, label="training loss")
plt.legend()
plt.show()