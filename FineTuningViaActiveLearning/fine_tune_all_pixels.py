import numpy as np
from torch.autograd import Variable
import torch
from model18 import GlassNetMod
from loss import ReflectionLoss
from dataset import AugmentedGSD, PanningAugmentedGSD
import csv
import torch.optim as optim
import warnings
import sys
from loss import lovasz_hinge
from torch.utils.data import DataLoader

# Ignore warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

# Writes a string to an output log file
def debugLog(msg):
    fi = open('debug_log.csv', 'a')
    writer = csv.writer(fi)
    writer.writerow([msg])
    fi.close()

# Are we using the CPU or GPU?
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define the model
path = "Model150"
model = GlassNetMod()
model.load_state_dict(torch.load(path, map_location='cpu'))
model.eval()
model = model.to(device)

# Get dataset file locations
if len(sys.argv) >= 2:
    gsd_path = sys.argv[1]
else:
    print("Please enter a path to the dataset.  Dataset path should be the Glass Surface Dataset (GSD) root directory.")
    quit()

#  Optimiser
ft_optimiser = optim.SGD(model.parameters(), lr=1e-5, weight_decay=5e-4, momentum=0.9)
baseline_optimiser = optim.SGD(model.parameters(), lr=1e-5, weight_decay=5e-4, momentum=0.9)

# Define loss
r_loss = ReflectionLoss()
r_loss = r_loss.to(device)

# Hyperparameters
epochs = 20
batch_size = 1
output_size = 384

# Set up the datasets
gsd = PanningAugmentedGSD(gsd_path + "GSD/train/", True)
dataloader = DataLoader(gsd, batch_size=batch_size, shuffle=True)
"""gsd_validation = PanningAugmentedGSD(gsd_path + "GSD/validation/", False)
dataloader_validation = DataLoader(gsd_validation, batch_size=2, shuffle=True)"""
gsd_original = AugmentedGSD(gsd_path + "GSD/train/", False)
dataloader_original = DataLoader(gsd_original, batch_size=batch_size, shuffle=True)

# Main training loop
for e in range(epochs):
    print("epoch : " + str(e))
    total_loss_ft = 0
    ft_count = 0
    total_loss_baseline = 0
    baseline_count = 0
    total_val_loss = 0

    ###############
    # FINE TUNING #
    ###############
    
    for data in dataloader:
        # Get data
        img1 = Variable(data[0])
        mask1 = Variable(data[1])
        img2 = Variable(data[3])
        mask2 = Variable(data[4])
        img3 = Variable(data[6])
        mask3 = Variable(data[7])
        trans_x = data[9]
        trans_y = data[10]
        np.random.seed()
        
        # Pick a random image from GSD
        debugLog("epoch : " + str(e) + ", frame " + str(ft_count) + " out of " + str(len(gsd)))
        print("epoch : " + str(e) + ", frame " + str(ft_count) + " out of " + str(len(gsd))) 

        # First forward pass on this random image
        img1 = img1.to(device)
        d0_img1, d1_img1, d2_img1, d3_img1, d4_img1, ref_img1 = model(img1.float())

        # Convert output to a binary segmentation map
        direct_output_1 = np.squeeze(d0_img1.cpu().detach().numpy())
        direct_output_1[direct_output_1 >= 0.5] = 1
        direct_output_1[direct_output_1 < 0.5] = 0

        # Second forward pass 
        img2[:,3,:,:] = torch.from_numpy(direct_output_1)
        img2 = img2.to(device)
        d0_img2, d1_img2, d2_img2, d3_img2, d4_img2, ref_img2 = model(img2.float())
        # Convert output to a binary segmentation map
        direct_output_2 = np.squeeze(d0_img2.cpu().detach().numpy())
        direct_output_2[direct_output_2 >= 0.5] = 1
        direct_output_2[direct_output_2 < 0.5] = 0

        # Calculate loss
        mask2 = mask2.to(device)
        gsd_loss = lovasz_hinge(d0_img2, mask2, True)
        gsd_loss += lovasz_hinge(d1_img2, mask2, True)
        gsd_loss += lovasz_hinge(d2_img2, mask2, True)
        gsd_loss += lovasz_hinge(d3_img2, mask2, True)
        gsd_loss += lovasz_hinge(d4_img2, mask2, True)

        # Backwards pass for video data
        ft_optimiser.zero_grad()
        gsd_loss.backward()
        ft_optimiser.step()
        total_loss_ft = total_loss_ft + (gsd_loss.item())

        # Third forward pass 
        img3[:,3,:,:] = torch.from_numpy(direct_output_2)
        img3 = img3.to(device)
        d0_img3, d1_img3, d2_img3, d3_img3, d4_img3, ref_img3 = model(img3.float())

        # Convert output to a binary segmentation map
        direct_output_3 = np.squeeze(d0_img3.cpu().detach().numpy())
        direct_output_3[direct_output_3 >= 0.5] = 1
        direct_output_3[direct_output_3 < 0.5] = 0

        # Calculate loss
        mask3 = mask3.to(device)
        gsd_loss = lovasz_hinge(d0_img3, mask3.float(), True)
        gsd_loss += lovasz_hinge(d1_img3, mask3.float(), True)
        gsd_loss += lovasz_hinge(d2_img3, mask3.float(), True)
        gsd_loss += lovasz_hinge(d3_img3, mask3.float(), True)
        gsd_loss += lovasz_hinge(d4_img3, mask3.float(), True)

        # Backwards pass for video data
        ft_optimiser.zero_grad()
        gsd_loss.backward()
        ft_optimiser.step()
        total_loss_ft = total_loss_ft + (gsd_loss.item())
        
        ft_count += 2

    #####################
    # BASELINE TRAINING #
    #####################

    debugLog("BASELINE TRAINING")
    for data in dataloader_original:

        debugLog("iteration" + str(baseline_count))
        # Get data
        img = Variable(data[0])
        mask = Variable(data[1])
        ref_gt = Variable(data[2])
        transf = data[3]

        img = img.to(device)
        mask = mask.to(device)
        ref_gt = ref_gt.to(device)

        # Forward pass
        
        d0, d1, d2, d3, d4, ref = model(img.float())
        debugLog("Forward pass")

        mask = mask.to(device)
        ref_gt = ref_gt.to(device)
        loss = r_loss(d0, d1, d2, d3, d4, ref, ref_gt, mask)
        debugLog("Loss Cacluated")

        # Backwards pass
        baseline_optimiser.zero_grad()
        loss.backward()
        baseline_optimiser.step() 
        debugLog("Backward pass")

        # Calculate loss
        total_loss_baseline = total_loss_baseline + loss.item()  
        if (baseline_count % 100 == 0):
            print("Epoch: " + str(e) + ",Iteration: " + str(baseline_count) + ", Loss : " + str(loss.item()) + ", Transformations : " + str(transf))
        baseline_count += 1

    #############################
    # CALCULATE VALIDATION LOSS #
    ############################# 
    # • To use this block of code you will need to create your own train/validation split for the dataset
    # • Uncomment validation dataloaders above to use this section
    """for data in dataloader_validation:
        # Get data
        img1 = Variable(data[0])
        mask1 = Variable(data[1])
        img2 = Variable(data[3])
        mask2 = Variable(data[4])
        img3 = Variable(data[6])
        mask3 = Variable(data[7])
        trans_x = data[9]
        trans_y = data[10]
        np.random.seed()

        # Forward pass on this random image
        img1 = img1.to(device)
        d0_img1, d1_img1, d2_img1, d3_img1, d4_img1, ref_img1 = model(img1.float())

        # Second forward pass 
        direct_output_1 = np.squeeze(d0_img1.cpu().detach().numpy())
        direct_output_1[direct_output_1 >= 0.5] = 1
        direct_output_1[direct_output_1 < 0.5] = 0
        img2[:,3,:,:] = torch.from_numpy(direct_output_1)
        img2 = img2.to(device)
        d0_img2, d1_img2, d2_img2, d3_img2, d4_img2, ref_img2 = model(img2.float())

        # Calculate loss 
        mask2 = mask2.to(device)
        loss_res = lovasz_hinge(d0_img2, mask2.float(), True)
        loss_res += lovasz_hinge(d1_img2, mask2.float(), True)
        loss_res += lovasz_hinge(d2_img2, mask2.float(), True)
        loss_res += lovasz_hinge(d3_img2, mask2.float(), True)
        loss_res += lovasz_hinge(d4_img2, mask2.float(), True)

        # Third forward pass 
        direct_output_2 = np.squeeze(d0_img2.cpu().detach().numpy())
        direct_output_2[direct_output_2 >= 0.5] = 1
        direct_output_2[direct_output_2 < 0.5] = 0
        img3[:,3,:,:] = torch.from_numpy(direct_output_2)
        img3 = img3.to(device)
        d0_img3, d1_img3, d2_img3, d3_img3, d4_img3, ref_img3 = model(img3.float())

        # Calculate loss 
        mask3 = mask3.to(device)
        ref_gt3 = ref_gt3.to(device)
        loss_res += lovasz_hinge(d0_img3, mask3.float(), True)
        loss_res += lovasz_hinge(d1_img3, mask3.float(), True)
        loss_res += lovasz_hinge(d2_img3, mask3.float(), True)
        loss_res += lovasz_hinge(d3_img3, mask3.float(), True)
        loss_res += lovasz_hinge(d4_img3, mask3.float(), True)
        total_val_loss += loss_res.item()"""

    # Save loss for this epoch into a csv file
    fi = open('loss_history_all_pixels.csv', 'a')
    writer = csv.writer(fi)
    writer.writerow([str(e), str(total_loss_ft / (ft_count * 2)), str(total_loss_baseline / baseline_count)])
    fi.close()

    # Save the model parameters every 2 epochs
    if (e + 1) % 2 == 0:
        torch.save(model.state_dict(), "Model" + str(e+1) + "-AllPix") 