import numpy as np
import cv2
import random
import torch
from skimage import transform
from torch.utils.data import Dataset
import os
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as nn
import csv


class ReflectionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, d0, d1, d2, d3, d4, r_glass, r_global, ground_truth, ignore=None):
        # Parmaeters:
        #   d0              :   final predicted glass mask
        #   d1, d2, d3, d4  :   predicted masks from different feature levels
        #   r_glass         :   predicted reflection
        #   r_global        :   ground truth reflection
        #   ground_truth    :   ground truth glass segmentation

        # Weighting of every component of the loss function (by default 1 as per the "Dont hit me! Glass Detection..." paper)
        weights = np.ones(6)

        # Define loss variables
        loss = 0
        mse_loss = nn.MSELoss()

        # Add dimension to ground truth mask
        ground_truth = torch.unsqueeze(ground_truth, 1)

        # Add reflection based loss
        loss = weights[0] * mse_loss(torch.mul(r_global, ground_truth), torch.mul(r_glass, ground_truth))

        # Add difference between predicted and ground truth segmentation masks
        loss += weights[1] * lovasz_hinge(d0, ground_truth, True, ignore)
        loss += weights[2] * lovasz_hinge(d1, ground_truth, True, ignore)
        loss += weights[3] * lovasz_hinge(d2, ground_truth, True, ignore)
        loss += weights[4] * lovasz_hinge(d3, ground_truth, True, ignore)
        loss += weights[5] * lovasz_hinge(d4, ground_truth, True, ignore)

        return loss


def debugLog(msg):
    # log
    fi = open('debug_log.csv', 'a')
    writer = csv.writer(fi)
    writer.writerow([msg])
    fi.close()


# All following code is taken from this URL:
# https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py
"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)    # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []    
        for i in range(C):
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)] # mean accross images if per_image
    return 100 * np.array(ious)


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        if ignore == None:
            loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                for log, lab in zip(logits, labels))
        else:
            loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ign.unsqueeze(0)))
                for log, lab, ign in zip(logits, labels, ignore))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore == None:
        return scores, labels
    else:
        ignore = ignore.view(-1)
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
         super(StableBCELoss, self).__init__()
    def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore == None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x
    
    
def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class AugmentedGSD(Dataset):
    def __init__(self, training_path):
        self.img_path = training_path + "image/" #"/Users/harveymannering/Documents/University/Project/GSD/GSD/train/image/"
        self.mask_path = training_path + "mask/" #"/Users/harveymannering/Documents/University/Project/GSD/GSD/train/mask/"
        self.reflections_path = training_path + "reflections/" #"/Users/harveymannering/Documents/University/Project/GSD/GSD/train/reflections/"
        self.filenames = os.listdir(self.reflections_path)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        # Define path
        filename = self.filenames[idx]
        filename = filename[:-4]

        # Load image
        
        img = cv2.imread(str(self.img_path + str(filename) + ".jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load reflection image
        reflection = cv2.imread(str(self.reflections_path + str(filename) + ".png"))
        reflection = cv2.cvtColor(reflection, cv2.COLOR_BGR2RGB)

        # Load mask
        mask = cv2.imread(str(self.mask_path + str(filename) + ".png"))
        mask = mask[:,:,0]

        # Normalize mask
        if np.max(mask) < 1e-6:
            mask = mask
        else:
            mask = mask / np.max(mask)

        # Random offset function
        randOffset = lambda scale: np.random.normal() * scale

        # Get image dimensions
        h,w,_ = img.shape

        # Randomize whether a homography or a crop is done to the mask
        transformationType = random.randint(0, 2)

        # Homography
        if transformationType <= 1:
            # Define point from (corners of the image)
            pts_src = np.array([[0, 0], 
                                [0, h], 
                                [w, h], 
                                [w, 0]])

            # Define points to map to (corners but slightly offset)
            s = int((w+h) * 0.025)
            pts_dst = np.array([[randOffset(s), randOffset(s)], 
                                [randOffset(s), h + randOffset(s)], 
                                [w + randOffset(s), h + randOffset(s)], 
                                [w + randOffset(s), randOffset(s)]])

            # Calculate homography H1
            H1, status = cv2.findHomography(pts_src, pts_dst)

            # Apply homography
            mask_augmented = cv2.warpPerspective(mask, H1, (w,h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0.5))

        # Crop image
        elif transformationType == 2:
            # Augment mask for previous frame 
            s = 25
            x_start = random.randint(0, s) #int(np.abs(randOffset(s)))
            y_start = random.randint(0, s) #int(np.abs(randOffset(s)))
            x_end = random.randint(0, s) #int(np.abs(randOffset(s)))
            y_end = random.randint(0, s) #int(np.abs(randOffset(s)))
            mask_augmented = mask[y_start:h-y_end, x_start:w-x_end]

            # Pick a direction to shift the frame in (which direction will give the largest shift)
            randomDirection =  np.argmin([x_end, y_end, x_start, y_start])
            # Left
            if randomDirection == 0:
                x_end += x_start
                x_start = 0   
            # Up
            elif randomDirection == 1:
                y_end += y_start
                y_start = 0
            # Right
            elif randomDirection == 2:
                x_start += x_end
                x_end = 0
            # Down
            elif randomDirection == 3:
                y_start += y_end
                y_end = 0

            # Crop mask and image and reflection
            mask = mask[y_start:h-y_end, x_start:w-x_end]
            img = img[y_start:h-y_end, x_start:w-x_end, :]
            reflection = reflection[y_start:h-y_end, x_start:w-x_end, :]


        # change the color space
        tmpImg = np.zeros((img.shape[0], img.shape[1], 3))
        img = img / np.max(img)
        if img.shape[2] == 1:
            tmpImg[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (img[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 2] = (img[:, :, 0] - 0.485) / 0.229
        else:
            tmpImg[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
            tmpImg[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225

        # Resize images
        output_size = 384
        img = transform.resize(tmpImg, (output_size, output_size), mode='constant')
        mask = transform.resize(mask, (output_size, output_size), mode='constant', order=0, preserve_range=True)
        mask_augmented = transform.resize(mask_augmented, (output_size, output_size), mode='constant', order=0, preserve_range=True)
        reflection = transform.resize(reflection, (output_size, output_size), mode='constant')

        # Stack the colour image with the augmented mask to create the input to the network
        stacked_img = np.zeros((output_size, output_size, 4))
        stacked_img[:,:,:3] = img
        stacked_img[:,:,3] = mask_augmented

        # Change dimensions to be PyTorch friendly
        stacked_img = np.transpose(stacked_img, (2, 0, 1))
        reflection = np.transpose(reflection, (2, 0, 1))

        # Show images
        """plt.imshow(mask)
        plt.show()
        plt.imshow(reflection)
        plt.show()
        plt.imshow(stacked_img[:,:,:3])
        plt.show()
        plt.imshow(stacked_img[:,:,3], cmap='gray')
        plt.show()"""

        return stacked_img, mask, reflection
