import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model import GlassNetMod
from loss import ReflectionLoss
from dataset import EvaluateGSD, AugmentedGSD
import csv
import sys
import numpy as np
import math

# Computer intersection of union for one image
def compute_iou(predict_mask, gt_mask):

    if np.sum(predict_mask) == 0 or np.sum(gt_mask) == 0:
        iou_ = 0
        return iou_

    n_ii = np.sum(np.logical_and(predict_mask, gt_mask))
    t_i = np.sum(gt_mask)
    n_ij = np.sum(predict_mask)

    iou_ = n_ii / (t_i + n_ij - n_ii)

    return iou_

# Calculate the pixel accuracy as: (# correct pixels) / (# total pixels)
def compute_pixel_acc(predict_mask, gt_mask):

    N_p = np.sum(gt_mask)
    N_n = np.sum(np.logical_not(gt_mask))

    TP = np.sum(np.logical_and(predict_mask, gt_mask))
    TN = np.sum(np.logical_and(np.logical_not(predict_mask), np.logical_not(gt_mask)))

    accuracy_ = (TP + TN) / (N_p + N_n)

    return accuracy_


# Compute f-measure/f-score for one image
def compute_f_score(predict_mask, gt_mask):
    beta = 0.3

    # Computer false/true positives/negatives
    N_p = np.sum(gt_mask)
    N_n = np.sum(np.logical_not(gt_mask))
    TP = np.sum(np.logical_and(predict_mask, gt_mask))
    TN = np.sum(np.logical_and(np.logical_not(predict_mask), np.logical_not(gt_mask)))
    FP = N_p - TP
    FN = N_n - TN

    precission =  TP / (TP + FP) 
    recall = TP / (TP + FN) 

    f = ((1+beta*beta)*precission*recall) / ((beta*beta*precission) + recall)

    return f

# Are we using the CPU or GPU?
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up the dataset
dataset_path = sys.argv[1]
dataset = EvaluateGSD(dataset_path + "GSD/test/")
bs = 1
dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

# Define loss
r_loss = ReflectionLoss()
r_loss = r_loss.to(device)


# Define the model
#path = "PATH TO MODEL .pth FILE SHOULD GO HERE"
model = GlassNetMod()
#model.load_state_dict(torch.load(path, map_location=device))
#model.eval()
model = model.to(device)
sumIoU = 0
sumPA = 0
sumF = 0
totalIoU = 0
totalPA = 0
totalF = 0

for data in dataloader:

    # Get data
    img = Variable(data[0])
    mask = data[1]
    mask /= np.max(mask.detach().numpy())
    img = img.to(device)

    # Forward pass
    d0, d1, d2, d3, d4, ref = model(img.float())

    direct_output = d0.cpu().detach().numpy()
    direct_output[direct_output >= 0.5] = 1
    direct_output[direct_output < 0.5] = 0

    # Calculate IoU
    for b in range(bs):
        
        IoU = compute_iou(np.squeeze(direct_output[b,:,:]), mask[b,:,:].detach().numpy())
        if not math.isnan(IoU):
            sumIoU += IoU
            totalIoU += 1
        
        PA = compute_pixel_acc(np.squeeze(direct_output[b,:,:]), mask[b,:,:].detach().numpy())
        if not math.isnan(PA):
            sumPA += PA
            totalPA += 1

        f = compute_f_score(np.squeeze(direct_output[b,:,:]), mask[b,:,:].detach().numpy())
        if not math.isnan(f):
            sumF += f
            totalF += 1
        
        print(IoU, PA, f)

# Save loss for this epoch into a csv file
f = open('evaluation_metrics.csv', 'a')
writer = csv.writer(f)
writer.writerow([str(sumIoU), str(totalIoU), str(sumPA), str(totalPA), str(sumF), str(totalF)])
f.close()