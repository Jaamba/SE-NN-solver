import torch
from torch import optim
from torch import nn
import numpy as np
import helper
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Uses the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# The set on which the SE will be solved will be like [-A, A]
A = 10
# Discretization size of the set
N = 256
t = np.linspace(-A, A, N)
dt = t[1] - t[0]

# Defines the neural network class
class SESolver(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, N+1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        E = x[:, 0]
        E = E.unsqueeze(1)
        phi = x[:, 1:]

        # Makes sure that the output function is normalized
        integral = torch.sqrt(dt * torch.sum(phi**2, dim=1, keepdim=True))
        phi = phi / integral

        return E, phi
    
# Loss function definition for the network
def criterion(E, phi, V):
    return torch.mean( (helper.hamiltonian(phi, V, dt) - E*phi)**2 )

# Build a feed-forward network. 
model = SESolver().to(device)

# Model optimizer to train the model
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Number of batches to generate
pols = 10000
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
    input = helper.random_function(BATCH_SIZE, N, device=device)

    for j in range(epoch):
        optimizer.zero_grad()

        # Forward press
        E, phi = model(input)

        # Calculates the loss
        loss = criterion(E, phi, input)

        # Bacward press
        loss.backward()
        optimizer.step()

# Example
V = torch.empty(N).to(device)
V[:] = 0
V[int(N/3):int(2*N/3)] = 1
V = V.unsqueeze(0)
print(V.shape)
E, phi = model(V)

t = np.linspace(-A, A, N)
plt.plot(t, V.squeeze().cpu().detach().numpy(), label="input")
print(phi.shape)
plt.plot(t, phi.squeeze().cpu().detach().numpy(), label="model")
print(E.item())
plt.legend()
plt.show()
