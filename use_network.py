import torch
from network import config
from network import network
from network import helper
from network import dataset
from network import training
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Uses the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: " + str(device))

# Finds the root folder and checkpoint path 
ROOT = Path(__file__).resolve().parent
statedict_path = ROOT / "network" / "data" / "checkpoint.pth"
testing_path = ROOT / "network" / "data" / "testingset.pt"

# Creates the model
model = network.FourierNet().to(device)

# Asks to train the network
if(input("Do you want to train the network? (y/N):") == 'y'):
    showInfo = False
    if(input("Do you want training loss to be plotted? (y/N):") == 'y'):
        showInfo = True

    print("Training network...")
    training.trainNetwork(showInfo)
    print("Training done successfully.")

# Loads the network if it exists 
if(Path(statedict_path).exists()):
    print("Loading model...")
    state_dict = torch.load(statedict_path)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded correctly")
else:
    print("State dict not found. The model parameters will be randomly assigned")

if(input("Do you want to test the network? (y/N):")== 'y'):

    # Generates the testing set if it doesn't exist
    if(Path(testing_path).exists() == False):
        print("Testing set not found. A new one will be generated:")

        dataset.generate_testing_set(testing_path, device=device, type=config.TESTING_TYPE)

    # Loads the testing set
    print("Loading testingset...")
    testset = torch.load( testing_path, map_location="cpu")
    print("Testing set loaded correctly")

    # Regenerates the set if the file does not match the config file
    if(testset["function"].shape != (config.TESTING_NUM_BATCHES, config.TESTING_BATCH_SIZE, config.N)
        or testset["phi"].shape != (config.TESTING_NUM_BATCHES, config.TESTING_BATCH_SIZE, config.N)
        or testset["E"].shape != (config.TESTING_NUM_BATCHES, config.TESTING_BATCH_SIZE, 1)):

        print("The testing set found in ", testing_path, " does not match the config file.")
        print("A new testing set will be generated:")

        # Creates and loads the new testing set
        dataset.generate_testing_set(testing_path, device=device, type=config.TESTING_TYPE)
        testset = torch.load( testing_path, map_location="cpu")
        print("New testset loaded correctly")

    # obtains the data from the testset
    func_set = testset["function"]
    phi_set = testset["phi"]
    E_set = testset["E"]

    # Begins testing
    func_score = 0
    E_score = 0
    for i in range(config.TESTING_NUM_BATCHES):

        # Takes the current batch
        f = func_set[i].to(device, non_blocking=True)
        phi = phi_set[i].to(device, non_blocking=True)
        E = E_set[i].to(device, non_blocking=True)

        # Calculates the model phi and E
        m_E, m_phi, _ = model(f)

        # Function score remains low if the model phi^2 is similar to phi^2
        func_score += torch.mean( (m_phi**2 - phi**2)**2 )
        E_score += torch.mean( (E - m_E)**2 )

        # Prints progress
        print(f"\rTesting network: {i+1}/{config.TESTING_NUM_BATCHES} ({100*(i+1)/config.TESTING_NUM_BATCHES:.1f}%)", end="", flush=True)

        # Makes a graph N_GRAPHS times
        if(i % (config.TESTING_NUM_BATCHES // config.N_GRAPHS) == 0 and config.PLOT_GRAPHS):
            # Choses which function to plot
            k = torch.randint(0, config.TESTING_BATCH_SIZE, (1,))

            # takes the functions to plot from the batch
            f_plot = f[k].squeeze().detach().cpu().numpy()
            phi_plot = (phi[k]**2).squeeze().detach().cpu().numpy()
            m_phi_plot = (m_phi[k]**2).squeeze().detach().cpu().numpy()

            t = np.linspace(-config.A, config.A, config.N)

            # Plots the graph
            plt.plot(t, f_plot, label="V")
            plt.plot(t, phi_plot, label="Phi^2 (fd)")
            plt.plot(t, m_phi_plot, label="Phi^2 (Model)")
            plt.legend()

            # Prints energy info
            print()
            print(f"The plotted graph has fd energy {E[k].squeeze().item():.6f}, while model energy is {m_E[k].squeeze().item():.6f}")
            print("Close the graph to continue testing")

            plt.show()
    print()


    # Takes the average between all batches
    func_score /= config.TESTING_NUM_BATCHES
    E_score /= config.TESTING_NUM_BATCHES

    print(f"Testing done successfully. scored: func_score: {func_score}, E_score: {E_score}")