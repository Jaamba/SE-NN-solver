import torch
from network import config
from network import network
from network import helper
from network import training
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Finds the root folder and checkpoint path 
ROOT = Path(__file__).resolve().parent
statedict_path = ROOT / "network" / "data" / "checkpoint.pth"

# Loads the model from the file
model = network.FourierNet()

# Asks to train the network
if(input("Do you want to train the network? (y/N):") == 'y'):
    showInfo = False
    if(input("Do you want training loss to be plotted? (y/N):") == 'y'):
        showInfo = True

    print("Training network...")
    training.trainNetwork(showInfo)
    print("Training done successfully.")


print("Loading model...")
state_dict = torch.load(statedict_path)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded correctly")

# Loads the model info from the config file
N = config.N
A = config.A
dt = 2*A/(N-1)

# Tests the model with a potential well
V = torch.empty(N)
V[:] = 0
V[int(40*N/100):int(60*N/100)] = -1
V = V.unsqueeze(0)
#V = dataset.random_gaussian_wells(1, N, max_wells=2, device="cpu")
E, phi, _ = model(V)

# Gets the teoric values of E and phi
Eteor, phiTeor = helper.solve_schrodinger(V, dt, 0)
Eteor2, phiTeor2 = helper.solve_schrodinger(V, dt, 1)

# Plots the results
t = np.linspace(-A, A, N)
plt.plot(t, V.squeeze().cpu().detach().numpy(), label="input")
plt.plot(t, phi.squeeze().cpu().detach().numpy(), label="model")
plt.plot(t, phiTeor[0].squeeze().cpu().detach().numpy(), label="teor")
plt.plot(t, phiTeor2[0].squeeze().cpu().detach().numpy(), label="teor2")
print("Model energy: " + str(E.item()))
print("Teoric energy: " + str(Eteor[0].item()))
print("Teoric energy 2: " + str(Eteor2[0].item()))
plt.legend()
plt.show()