import torch
from torch import optim
import network.network as network
import matplotlib.pyplot as plt
import network.dataset as dataset
from pathlib import Path
import network.config as config

# Loss function definition for the network training
def criterion(E, phi, Hphi):
    #residual = torch.mean(
    #    (Hphi-E*phi)**2
    #)

    return torch.mean(E) 

# Trains the network
def trainNetwork(plotTraining = False):

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
    statedict_file = Path(statedict_path)
    if(not statedict_file.exists()):
        print("Statedict file not found. A new network will be generated.")
    elif(input("Do you want to load previously trained network? (y/N)").lower() == "y"):
        print("Loading existing network...")
        state_dict = torch.load(statedict_path)
        model.load_state_dict(state_dict)
        print("Network loaded correctly")

    # Model optimizer to train the model
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Loads the dataset
    trainingset_file = Path(trainingset_path)
    if(trainingset_file.exists()):
        print("Dataset found in ", trainingset_path)
        print("Loading dataset...")
        trainingset = torch.load( trainingset_path, map_location="cpu")

        # Checks if the shape is the same as the one in the config
        if( trainingset.shape != (config.NUM_BATCHES, config.BATCH_SIZE, config.N) ):
            print("The dataset found in ", trainingset_path, " does not match the data in" \
            " the configuration file. A new dataset will be generated:")

            type = input("Choose dataset type: (mixed/smooth/well/poly) ")
            dataset.generate_training_set(trainingset_path, device="cuda", type=type)
    else:
        # Generates a new trainingset 
        print("Dataset not found. A new dataset will be generated.")
        type = input("Choose dataset type: (mixed/smooth/well/poly) ")
        dataset.generate_training_set(trainingset_path, device="cuda", type=type)

        print("Loading dataset...")
        trainingset = torch.load( trainingset_path, map_location="cpu")
    print("Dataset loaded corretctly")

    # Imports information from config file
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
    if(plotTraining):
        plt.plot(train_losses, label="training loss")
        plt.legend()
        plt.show()