from cmath import nan
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, DrawingArea, HPacker, VPacker, OffsetImage
import matplotlib.patches as mpatches
import numpy as np
import cv2
from torch.autograd import Variable
import io
import torch
from model18 import GlassNetMod
from loss import ReflectionLoss
from dataset import PreProcessRGDMImage, Tensor2Img
from dataset import AugmentedGSD, PanningAugmentedGSD
import matplotlib.pyplot as plt
from skimage import transform
import os
import csv
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import warnings
from os.path import exists
import time
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

def RegionGrowth(img, seeds):
    # 8 Neighborhood 
    directs = [(-1,-1), (0,-1), (1,-1), (1,0), (1,1), (0,1),(-1,1),(-1,0)]

    # Initalize variables
    visited = np.zeros(shape=(img.shape), dtype=np.uint8)
    out_img = np.zeros_like(img)
    h, w = img.shape

    # Loop over all orginal seeds
    while len(seeds):
        seed = seeds.pop(0)
        x = seed[1]
        y = seed[0]
        
        # visit point (x,y)
        visited[y][x] = 1
        for direct in directs:
            cur_x = x + direct[0]
            cur_y = y + direct[1]

            # illegal 
            if cur_x <0 or cur_y<0 or cur_x >= w or cur_y >=h:
                continue

            # Not visited and belong to the same target 
            if (not visited[cur_y][cur_x]) and (img[cur_y][cur_x] > 0):
                out_img[cur_y][cur_x] = 1
                visited[cur_y][cur_x] = 1
                seeds.append((cur_y,cur_x))

    return out_img

# Get the locations of the few remaining non zero pixels in the mask
def GetPixelLocations(mask):    
    max_idx = []
    output_size = 384
    mask_flat = mask.flatten().tolist()
    no_ones = mask_flat.count(1)
    for i in range(no_ones):
        idx = mask_flat.index(1)
        mask_flat[idx] = 0
        x = int(idx / output_size)
        y = idx % output_size
        max_idx.append((x,y))
    return max_idx

def PickFlickeringPixel(mask):
    # Define loop variables
    kernel = np.ones((3,3), np.uint8)
    it = 0

    prev_masks = []
    # Erode mask until there are no pixel left
    while np.max(mask) > 0:
        # Save last iterations mask
        prev_masks.append(mask.copy())

        # Erosion - test every pixel with a structuring element (kernel), if it is a fit then keep the pixel 
        # as a 1, if it is a hit then set the pixel to zero
        mask = cv2.erode(mask, kernel, iterations=1)
        it += 1

    if it >= 2:
        # Get the locations of the few remaining pixels from the penultimate eroded mask
        max_idx = GetPixelLocations(prev_masks[-1])
        seed = [max_idx[np.random.randint(len(max_idx))]]

        # Perform region growth on an uneroded mask
        uneroded_mask = prev_masks[-2]
        reg_growth_mask = RegionGrowth(uneroded_mask, seed.copy())

        # Get the locations of the few remaining pixels from the penultimate eroded mask
        max_idx = GetPixelLocations(reg_growth_mask)
        
        # Pick a random pixel that remains and return it
        return it, max_idx[np.random.randint(len(max_idx))]
    else:
        return 0, ()

def QuantifyFlicker(direct_output1, direct_output2, trans_x, trans_y):
    # Translate first output
    M = np.float32([[1, 0, int(trans_x.item() / 2)],
                    [0, 1, int(trans_y.item() / 2)],
                    [0, 0, 1]])
    direct_output1[:,:] = cv2.warpPerspective(direct_output1[:,:], M, (direct_output1.shape[0], direct_output1.shape[1]), borderMode=cv2.BORDER_CONSTANT, borderValue=-10)

    # Calculate difference between masks
    problem_pixels = direct_output1 - direct_output2
    problem_pixels[problem_pixels == -1] = 1
    problem_pixels[problem_pixels >= 0.5] = 1
    problem_pixels[problem_pixels != 1] = 0

    return problem_pixels, np.sum(problem_pixels[:,:])

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

# Optimisers
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
dataset_path = sys.argv[1]
gsd = PanningAugmentedGSD(gsd_path + "GSD/train/", True)
dataloader = DataLoader(gsd, batch_size=batch_size, shuffle=True)
"""gsd_validation = PanningAugmentedGSD(gsd_path + "GSD/validation/", False)
dataloader_validation = DataLoader(gsd_validation, batch_size=1, shuffle=True)"""
gsd_original = AugmentedGSD(gsd_path + "GSD/train/", False)
dataloader_original = DataLoader(gsd_original, batch_size=batch_size, shuffle=True)

# Create databases for this run
problem_database = np.zeros((2,384,384,2710), dtype=np.uint8)

for e in range(epochs):
    print("epoch : " + str(e))
    total_loss_ft = 0
    ft_count = 0
    total_loss_baseline = 0
    baseline_count = 0
    total_val_loss = 0
    query_selection_count = 0

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
        idx = data[11]
        np.random.seed()
        
        # Pick a random image from GSD
        debugLog("epoch : " + str(e) + ", frame " + str(ft_count) + " out of " + str(len(gsd)))
        print("epoch : " + str(e) + ", frame " + str(ft_count) + " out of " + str(len(gsd))) 

        # First forward pass on this random image
        img1 = img1.to(device)
        d0_img1, d1_img1, d2_img1, d3_img1, d4_img1, ref_img1 = model(img1.float())
        # Convert output to binary mask
        direct_output_1 = np.squeeze(d0_img1.cpu().detach().numpy())
        direct_output_1[direct_output_1 >= 0.5] = 1
        direct_output_1[direct_output_1 < 0.5] = 0

        # Second forward pass 
        img2[:,3,:,:] = torch.from_numpy(direct_output_1)
        img2 = img2.to(device)
        d0_img2, d1_img2, d2_img2, d3_img2, d4_img2, ref_img2 = model(img2.float())
        
        # Convert output to binary mask
        direct_output_2 = np.squeeze(d0_img2.cpu().detach().numpy())
        direct_output_2[direct_output_2 >= 0.5] = 1
        direct_output_2[direct_output_2 < 0.5] = 0

        # Translate first output
        problem_pixels, total_problem_pixels = QuantifyFlicker(direct_output_1, direct_output_2, trans_x, trans_y)
        query_selection_count += total_problem_pixels

        # Acquasition function - Choose random pixel equal to the number of flickering pixels
        pixels_of_interest = []
        for i in range(int(total_problem_pixels)):
            pixels_of_interest.append((np.random.randint(384), np.random.randint(384)))
                
        # Figure out ground truth and an ignore mask for GSD pixels
        for p in range(len(pixels_of_interest)):
            problem_database[0, pixels_of_interest[p][1], pixels_of_interest[p][0], idx] = 1
        
        all_problem_pixels = problem_database[0, :, :, idx]
        all_problem_pixels[all_problem_pixels > 0.5] = 1
        problem_database[0, :, :, idx] = all_problem_pixels
        
        gsd_ground_truth1 = torch.zeros((batch_size,1,384,384))
        gsd_ground_truth1[0,0,:,:] = mask3 * torch.from_numpy(all_problem_pixels)
        gsd_ignore1 = torch.zeros((batch_size,1,384,384))
        gsd_ignore1[0,0,:,:] = torch.abs(1 - mask3) * torch.from_numpy(all_problem_pixels)

        # Calculte loss for these GSD dataset pixels
        gsd_ignore1 = gsd_ignore1.to(device)
        gsd_ground_truth1 = gsd_ground_truth1.to(device)
        mask2 = mask2.to(device)
        gsd_loss = lovasz_hinge(d0_img2, gsd_ground_truth1.float(), True, gsd_ignore1.float())
        gsd_loss += lovasz_hinge(d1_img2, gsd_ground_truth1.float(), True, gsd_ignore1.float())
        gsd_loss += lovasz_hinge(d2_img2, gsd_ground_truth1.float(), True, gsd_ignore1.float())
        gsd_loss += lovasz_hinge(d3_img2, gsd_ground_truth1.float(), True, gsd_ignore1.float())
        gsd_loss += lovasz_hinge(d4_img2, gsd_ground_truth1.float(), True, gsd_ignore1.float())

        # Backwards pass for video data
        ft_optimiser.zero_grad()
        gsd_loss.backward()
        ft_optimiser.step()

        # Save loss for whole image
        gsd_loss = lovasz_hinge(d0_img2, mask2, True)
        gsd_loss += lovasz_hinge(d1_img2, mask2, True)
        gsd_loss += lovasz_hinge(d2_img2, mask2, True)
        gsd_loss += lovasz_hinge(d3_img2, mask2, True)
        gsd_loss += lovasz_hinge(d4_img2, mask2, True)
        total_loss_ft = total_loss_ft + (gsd_loss.item())

        # Third forward pass 
        img3[:,3,:,:] = torch.from_numpy(direct_output_2)
        img3 = img3.to(device)
        d0_img3, d1_img3, d2_img3, d3_img3, d4_img3, ref_img3 = model(img3.float())
        # Convert output to binary mask
        direct_output_3 = np.squeeze(d0_img3.cpu().detach().numpy())
        direct_output_3[direct_output_3 >= 0.5] = 1
        direct_output_3[direct_output_3 < 0.5] = 0

        problem_pixels, total_problem_pixels = QuantifyFlicker(direct_output_2, direct_output_3, trans_x, trans_y)
        query_selection_count += total_problem_pixels

        # Acquasition function - Choose random pixel equal to the number of flickering pixels
        pixels_of_interest = []
        for i in range(int(total_problem_pixels)):
            pixels_of_interest.append((np.random.randint(384), np.random.randint(384)))
   
        # Figure out ground truth and an ignore mask for GSD pixels
        for p in range(len(pixels_of_interest)):
            problem_database[1, pixels_of_interest[p][0], pixels_of_interest[p][1], idx] = 1
        
        all_problem_pixels = problem_database[1, :, :, idx]
        all_problem_pixels[all_problem_pixels > 0.5] = 1
        problem_database[1, :, :, idx] = all_problem_pixels
        
        gsd_ground_truth2 = torch.zeros((batch_size,1,384,384))
        gsd_ground_truth2[0,0,:,:] = mask3 * torch.from_numpy(all_problem_pixels)
        gsd_ignore2 = torch.zeros((batch_size,1,384,384))
        gsd_ignore2[0,0,:,:] = torch.abs(1 - mask3) * torch.from_numpy(all_problem_pixels)  

        # Calculte loss for these GSD dataset pixels
        gsd_ignore2 = gsd_ignore2.to(device)
        gsd_ground_truth2 = gsd_ground_truth2.to(device)
        mask3 = mask3.to(device)
        gsd_loss = lovasz_hinge(d0_img3, gsd_ground_truth2.float(), True, gsd_ignore2.float())
        gsd_loss += lovasz_hinge(d1_img3, gsd_ground_truth2.float(), True, gsd_ignore2.float())
        gsd_loss += lovasz_hinge(d2_img3, gsd_ground_truth2.float(), True, gsd_ignore2.float())
        gsd_loss += lovasz_hinge(d3_img3, gsd_ground_truth2.float(), True, gsd_ignore2.float())
        gsd_loss += lovasz_hinge(d4_img3, gsd_ground_truth2.float(), True, gsd_ignore2.float())

        # Backwards pass for video data
        ft_optimiser.zero_grad()
        gsd_loss.backward()
        ft_optimiser.step()

        # Save loss for whole image
        gsd_loss = lovasz_hinge(d0_img3, mask3, True)
        gsd_loss += lovasz_hinge(d1_img3, mask3, True)
        gsd_loss += lovasz_hinge(d2_img3, mask3, True)
        gsd_loss += lovasz_hinge(d3_img3, mask3, True)
        gsd_loss += lovasz_hinge(d4_img3, mask3, True)
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
        ref_gt1 = Variable(data[2])
        img2 = Variable(data[3])
        mask2 = Variable(data[4])
        ref_gt2 = Variable(data[5])
        img3 = Variable(data[6])
        mask3 = Variable(data[7])
        ref_gt3 = Variable(data[8])
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

        # Binaries output
        direct_output_2 = np.squeeze(d0_img2.cpu().detach().numpy())
        direct_output_2[direct_output_2 >= 0.5] = 1
        direct_output_2[direct_output_2 < 0.5] = 0

        # Translate first output
        problem_pixels, total_problem_pixels = QuantifyFlicker(direct_output_1, direct_output_2, trans_x, trans_y)

        # Calculate loss 
        mask2 = mask2.to(device)
        ref_gt2 = ref_gt2.to(device)
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

        # Binaries output
        direct_output_3 = np.squeeze(d0_img3.cpu().detach().numpy())
        direct_output_3[direct_output_3 >= 0.5] = 1
        direct_output_3[direct_output_3 < 0.5] = 0

        # Translate first output
        problem_pixels, total_problem_pixels = QuantifyFlicker(direct_output_2, direct_output_3, trans_x, trans_y)

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
    fi = open('loss_history_flicker_random.csv', 'a')
    writer = csv.writer(fi)
    writer.writerow([str(e), str(total_loss_ft / (ft_count * 2)), str(ft_count), str(query_selection_count), str(total_loss_baseline / baseline_count), str(np.sum(problem_database))])
    fi.close()

    # Save the model parameters every 2 epochs
    if (e + 1) % 2 == 0:
        torch.save(model.state_dict(), "Model" + str(e+1) + "-FlickRand")