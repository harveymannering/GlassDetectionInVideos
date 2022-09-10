import numpy as np
import cv2
import random
from skimage import transform
from torch.utils.data import Dataset
import os
from imgaug import augmenters as iaa
import imgaug.augmenters.flip as flip
from os.path import exists

def ChangeColourSpace(img):
    # Change the color space (preprocessing)
    try:
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
        return tmpImg
    except:
        quit()

# Normalizes, scale and changes the dimension order for an RGD image
def PreProcessRGDImage(img):
    output_size = 384
    img = ChangeColourSpace(img)
    img = transform.resize(img, (output_size, output_size), mode='constant')
    img = np.transpose(img, (2, 0, 1))
    img = img[None, ...]
    img = img.astype(np.float32)
    return img

# Prepares a (red, green, blue, mask) input
def PreProcessRGDMImage(img, mask):
    # Recolour, resize and change dimensions
    output_size = 384
    img = ChangeColourSpace(img)
    img = transform.resize(img, (output_size, output_size), mode='constant')
    mask = transform.resize(mask, (output_size, output_size), mode='constant')
    img = np.transpose(img, (2, 0, 1))

    # stack rbg image with mask
    rgbm = np.zeros((1, 4, output_size, output_size))
    rgbm[0,:3,:,:] = img
    rgbm[0,3,:,:] = mask[:, :, 0]
    rgbm = rgbm.astype(np.float32)

    return rgbm


def Tensor2Img(tensor):
     return np.transpose(tensor[0,:,:,:].detach().numpy(), (1, 2, 0))

## Transformation functions
def RandomZoomMask(img):
    zoom = iaa.Crop(percent=(0, 0.25)) # random crop
    return zoom(image=img)

def RandomZoomImage(img, mask, ref):
    zoom = iaa.Crop(percent=(0, 0.25)) # random crop
    zoom = zoom.to_deterministic()
    return zoom(image=img), zoom(image=mask), zoom(image=ref)

# Uses imgaug library to to perform a minor perspective transform
def MinorTransformIA(img, lower=0.0, upper=0.1):
    persp = iaa.PerspectiveTransform(scale=(lower, upper), keep_size=True)
    return persp(image=img)

def RandomPanMotion1(img, mask, reflection):
    # Get image dimensions
    h,w,_ = img.shape

    # Augment mask for previous frame 
    s = 50
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

    return img, mask, reflection, mask_augmented

# Uses imgaug library to to perform a minor perspective transform
def MajorTransformIA(img):
    # Random how the image is mirrored
    randomNum = np.random.randint(3)

    # Mirror image horizontally
    if randomNum % 2 == 0:
        img = flip.flipud(img)

    # Mirror image vertially   
    if randomNum > 0:
        img = flip.fliplr(img)

    return img

def EmptyMask(dims, also_full_masks=False):
    # Returns either a mask of all zeros or all ones
    if also_full_masks == True:
        return np.random.randint(2) * np.ones(dims)
    # Returns a mask with only zeros in it
    else:
        return np.ones(dims)

# Sigma can a single value or a tuple specifying the range of sigmas to be used
def ApplyBlur(img, sig=(0.0, 8.0)):
    blur = iaa.GaussianBlur(sigma=sig)
    return blur(image=img)

def augment_image(img, colour_jitter, grey_scale, blur):
    h,w,c = img.shape 

    # Add noise to different noise channels depending on augmentation
    for cj in colour_jitter:
        if cj != 0:
            np.random.seed(grey_scale + blur + cj)
            noise = np.random.randint(0,25, (h, w)) # design jitter/noise here
            zitter = np.zeros_like(img)
            zitter[:, :, (cj-1) % 3] = noise  
            img = cv2.add(img, zitter)

    # Convert to grayscale
    if grey_scale:
        grey_img = np.mean(img, axis=2)
        img[:,:,0] = grey_img
        img[:,:,1] = grey_img
        img[:,:,2] = grey_img
        
    # Blur the image
    if blur > 1 and blur % 2 == 1:
        img = cv2.blur(img, (blur,blur), blur / 3)
    
    # Reset seed
    np.random.seed()
    
    return img


class AugmentedGSD(Dataset):
    def __init__(self, training_path, includes_reflections = False):
        # Load filenames of the dataset
        self.img_path = training_path + "image/" 
        self.mask_path = training_path + "mask/" 
        self.reflections_path = training_path + "reflections/" 
        if includes_reflections:
            self.filenames = os.listdir(self.reflections_path)
        else:
            self.filenames = os.listdir(self.img_path)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        # Define path
        filename = self.filenames[idx]
        filename = filename[:-4]

        # Define the image size that the network takes as an input
        output_size = 384

        # Load image
        img = cv2.imread(str(self.img_path + str(filename) + ".jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     
        if exists(str(self.mask_path + str(filename) + ".png")):
            # Load mask
            mask = cv2.imread(str(self.mask_path + str(filename) + ".png"))
            mask = mask[:,:,0]
        else:
            # Negative examples have no mask files, so just create empty masks/reflections
            mask = np.zeros(output_size, output_size)
        
        if exists(str(self.reflections_path + str(filename) + ".png")):
            # Load reflection image
            reflection = cv2.imread(str(self.reflections_path + str(filename) + ".png"))
            reflection = cv2.cvtColor(reflection, cv2.COLOR_BGR2RGB)

        else:
            # Negative examples have no mask files, so just create empty masks/reflections
            reflection = np.zeros((384,384,3))

        # Normalize mask
        if np.max(mask) < 1e-6:
            mask = mask
        else:
            mask = mask / np.max(mask)

        # Mirror half of the images
        mirror = random.randint(0, 1)
        if mirror == 1:
            img = flip.fliplr(img)
            mask = flip.fliplr(mask)
            reflection = flip.fliplr(reflection)

        # Randomize whether a minor homography is apply to the mask, major homography is applied to the mask, or an empty mask is used
        transformationType = random.randint(0, 11)   
        if transformationType in (0,1,2):
            mask_augmented = MinorTransformIA(mask)
        elif transformationType == 3:
            img, mask, reflection, mask_augmented = RandomPanMotion1(img, mask, reflection)
        elif transformationType == 4:
            mask_augmented = mask
            img, mask, reflection = RandomZoomImage(img, mask, reflection)
        elif transformationType == 5:
            mask_augmented = RandomZoomMask(mask)
        elif transformationType in (6,7,8):
            mask_augmented = EmptyMask((output_size,output_size), True)
        elif transformationType in (9,10,11):
            mask_augmented = MajorTransformIA(mask)

        # Blur half of the masks
        mirror = random.randint(0, 1)
        if mirror == 1:
            mask_augmented = ApplyBlur(mask_augmented)
           
        # change the color space
        tmpImg = ChangeColourSpace(img)

        # Resize images
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

        return stacked_img, mask, reflection, transformationType

class EvaluateGSD(Dataset):
    def __init__(self, training_path):
        # Load filenames of the dataset
        self.img_path = training_path + "image/" 
        self.mask_path = training_path + "mask/" 
        self.filenames = os.listdir(self.img_path)
        self.reflections_path = training_path + "reflections/"

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        # Define path
        filename = self.filenames[idx]
        filename = filename[:-4]

        # Define the image size that the network takes as an input
        output_size = 384

        # Load image
        img = cv2.imread(str(self.img_path + str(filename) + ".jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     
        if exists(str(self.mask_path + str(filename) + ".png")):
            # Load mask
            mask = cv2.imread(str(self.mask_path + str(filename) + ".png"))
            mask = mask[:,:,0]
        else:
            # Negative examples have no mask files, so just create empty masks/reflections
            mask = np.zeros(output_size, output_size)

        if exists(str(self.reflections_path + str(filename) + ".png")):
            # Load reflection image
            reflection = cv2.imread(str(self.reflections_path + str(filename) + ".png"))
            reflection = cv2.cvtColor(reflection, cv2.COLOR_BGR2RGB)

        else:
            # Negative examples have no mask files, so just create empty masks/reflections
            reflection = np.zeros((384,384,3))

        # Normalize mask
        if np.max(mask) < 1e-6:
            mask = mask
        else:
            mask = mask / np.max(mask)

        # change the color space
        tmpImg = ChangeColourSpace(img)

        # Resize images
        img = transform.resize(tmpImg, (output_size, output_size), mode='constant')
        mask = transform.resize(mask, (output_size, output_size), mode='constant', order=0, preserve_range=True)
        mask_augmented = np.zeros_like(mask)

        # Stack the colour image with the augmented mask to create the input to the network
        stacked_img = np.zeros((output_size, output_size, 4))
        stacked_img[:,:,:3] = img
        stacked_img[:,:,3] = mask_augmented

        # Change dimensions to be PyTorch friendly
        stacked_img = np.transpose(stacked_img, (2, 0, 1))

        reflection = transform.resize(reflection, (output_size, output_size), mode='constant')
        reflection = np.transpose(reflection, (2, 0, 1))

        return stacked_img, mask, reflection