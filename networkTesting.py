import torch
import network
import matplotlib.pyplot as plt
import numpy as np
import network.helper as helper
import network.dataset as dataset

# Loads the model from the file
model = network.FourierNet()
state_dict = torch.load('checkpoint.pth')
model.load_state_dict(state_dict)
model.eval()

# Loads the model info
N = network.N
dt = network.dt
A = network.A

# Tests the model
V = torch.empty(N)
V[:] = 0
V[int(40*N/100):int(60*N/100)] = -1
V = V.unsqueeze(0)
V = dataset.random_gaussian_wells(1, N, max_wells=2, device="cpu")
E, phi, _ = model(V)
Eteor, phiTeor = helper.solve_schrodinger(V, dt, 0)
Eteor2, phiTeor2 = helper.solve_schrodinger(V, dt, 1)

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