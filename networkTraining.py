import torch
from torch import optim
import network
import helper
import torch.nn.functional as F

# Uses the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Build the network
model = network.SESolver().to(device)

# Loss function definition for the network
def criterion(E, phi, Hphi):
    # Minimizing this will make sure that phi is a solution. Only considers inner values
    mse = torch.mean(
        (Hphi[:,1:-1] - E*phi[:,1:-1])**2
    )

    return mse + torch.mean(E)

# Model optimizer to train the model
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Number of batches to generate
pols = 10000
# batch size
BATCH_SIZE = 512
# Number of epochs
epoch = 3

# Makes sure to be in training mode
model.train()
for i in range(pols):

    # Prints current training progress
    if i % (pols/100) == 0:
        print("Current progress:", i/pols*100, "%") 

    # Input values for the network
    input = helper.random_function(BATCH_SIZE, network.N, device=device, sigma=36)

    for j in range(epoch):
        optimizer.zero_grad()

        # Forward press
        E, phi, Hphi = model(input)

        # Calculates the loss
        loss = criterion(E, phi, Hphi)

        # Bacward press
        loss.backward()
        optimizer.step()

# Saves the trained model
torch.save(model.state_dict(), 'checkpoint.pth')