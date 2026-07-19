import torch
import network
import matplotlib.pyplot as plt

# Loads the model from the file
model = network.SESolver()
state_dict = torch.load('checkpoint.pth')
model.load_state_dict(state_dict)
model.eval()

# Loads the model info
N = network.N
dt = network.dt

# Tests the model
V = torch.empty(N)
V[:] = 0
V[int(5*N/10):int(6*N/10)] = -1
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
