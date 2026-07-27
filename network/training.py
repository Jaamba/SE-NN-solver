import torch
from torch import optim
import network
import matplotlib.pyplot as plt
import dataset
from pathlib import Path
import config

# Uses the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: " + str(device))

# Finds the root folder and trainingset path 
ROOT = Path(__file__).resolve().parent.parent
trainingset_path = ROOT / "network" / "data" / "trainingset.pt"
statedict_path = ROOT / "network" / "data" / "checkpoint.pth"

# Build the network
model = network.FourierNet().to(device)

# Asks to load previously trained network  if available
if(input("Do you want to load previously trained network? (y/N)").lower() == "y"):
    state_dict = torch.load(statedict_path)
    model.load_state_dict(state_dict)

# Loss function definition for the network
def criterion(E, phi, Hphi):
    residual = torch.mean(
        (Hphi-E*phi)**2
    )

    return torch.mean(E) + 0.1*residual

# Model optimizer to train the model
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Loads the dataset
print("Loading dataset...")
trainingset = torch.load( trainingset_path, map_location="cpu")
print("Dataset loaded corretctly")

# Imports information from config file
BATCH_SIZE = config.BATCH_SIZE
NUM_BATCHES = config.NUM_BATCHES
epoch = config.EPOCH

# Losses for each epoch
train_losses = []

# Makes sure to be in training mode
model.train()
for j in range(epoch):

    # Shuffles the dataset
    dataset.shuffleDataset(trainingset)
    running_loss = 0

    for i, batch in enumerate(trainingset):
        optimizer.zero_grad()

        # Moves the batches to the used device
        batch = batch.to(device, non_blocking=True)

        # Forward press
        E, phi, Hphi = model(batch)

        # Calculates the loss
        loss = criterion(E, phi, Hphi)
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
torch.save(model.state_dict(), statedict_path)

# Prints information about the training progress
plt.plot(train_losses, label="training loss")
plt.legend()
plt.show()