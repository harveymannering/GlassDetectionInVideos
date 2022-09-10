import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model18 import GlassNetMod
from loss import ReflectionLoss
from dataset import AugmentedGSD
import csv
import sys

# Are we using the CPU or GPU?
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
#path = "PATH TO PRETRAINED MODEL WOULD GO HERE"
model = GlassNetMod()
#model.load_state_dict(torch.load(path))
#model.eval()
model = model.to(device)


# Define loss
r_loss = ReflectionLoss()
r_loss = r_loss.to(device)

# Initalize history csv file
f = open('loss_history.csv', 'w')
writer = csv.writer(f)
writer.writerow(["Epoch", "Total Loss"])
f.close()

# Hyperparameters
num_epochs = 500

# Set up the dataset
dataset_path = sys.argv[1]
dataset = AugmentedGSD(dataset_path + "GSD/train/")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Optimiser
optimiser = optim.SGD(model.parameters(), lr=0.00125, weight_decay=5e-4, momentum=0.9)

# Main training loop
for epoch in range(num_epochs):
    total_loss = 0
    iterations = 0
    for data in dataloader:

        # Get data
        img = Variable(data[0])
        mask = Variable(data[1])
        ref_gt = Variable(data[2])
        transf = data[3]

        # Move data to GPU or CPU
        img = img.to(device)
        mask = mask.to(device)
        ref_gt = ref_gt.to(device)

        # Forward pass
        d0, d1, d2, d3, d4, ref = model(img.float())
        loss = r_loss(d0, d1, d2, d3, d4, ref, ref_gt, mask)

        # Backwards pass
        optimiser.zero_grad()
        loss.backward()
        optimiser.step() 

        # Calculate loss
        total_loss = total_loss + loss.item()  
        if (iterations % 50 == 0):
            print("Epoch: " + str(epoch) + ",Iteration: " + str(iterations) + ", Loss : " + str(loss.item()) + ", Transformations : " + str(transf))
        iterations += 1

    # Calculate and display loss for whole epoch
    print("Epoch " + str(epoch))
    print(" Loss:" + str(total_loss))

    # Save loss for this epoch into a csv file
    f = open('loss_history.csv', 'a')
    writer = csv.writer(f)
    writer.writerow([str(epoch), str(total_loss)])
    f.close()

    # Save the model parameters every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), "Model" + str(epoch+1))

