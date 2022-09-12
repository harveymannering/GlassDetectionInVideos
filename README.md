# Glass Detection in Videos

MSc Project for Computer Graphics, Vision, and Imaging at UCL by Harvey Mannering

## Overview

This project aims to extend the glass detection model presented by Jiaying Lin et al. [^1] in to video.  We call this network GSDNet4Video.  This is done mainly using techniques found in Google's "Mobile Real-time Video Segmentation" blog[^2].  This repo also explores methods for fine tuning GSDNet4Video using active learning. We use code from the following codebases:
- https://github.com/NoelShin/PixelPick
- https://jiaying.link/cvpr2021-gsd/
- https://github.com/kshitijkumbar/LearningBasedMethods/blob/45a031df5c1e2fd7f0c6f377c1265f3540736be2/resNet/resnet.py
- https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py

## GSDNet4Video

Using the propagation-based techniques, we can convert GSDNet [^1] into a video model.  We call this new model GSDNet4Video.  While GSDNet has three input channels (RGB), GSDNet4Video has four.  The addition channel takes the previous frames mask.  In most videos, consecutive frames will be similar.  As a result, consecutive masks should also be similar.  Therefore, by using in the previous frames mask as an input, we are providing our network with a strong clue as to how the current frame's mask should look.  The issue is that we cannot train the network on a previous frames mask because we do not have that data.  There are no video dataset from glass detection.  However, by performing minor transformations on the masks that we do have, we can simulate a perspective shift. These slight change match what we would expect to see in a real video.  In this way, we can train a video model using single image datasets.

![Network drawio](https://user-images.githubusercontent.com/60523103/188334935-03532005-bb17-49d4-8309-1119bc8af6fd.png)

## Active Learning
In this portion of the project we aim to fine tune the baseline GSDNet4Video model already developed using active learning.  We draw heavily on the framework used in PixelPick [^3] which can increase segmentation performance by training on just a pixel labels per image.  We also explore a new query strategy based on flicker between frames as a mean of improving temporal stability.  Flicker can be defined an object changing classification from one frame to the next and can be seen as type of temporal uncertainty.  While there is evidence to suggest that training on flickering can effectively reduce the labelling burden, our findings are not conclusive. 

## Code

Before running the files here, you should download the GSD dataset.  This can be found at https://jiaying.link/cvpr2021-gsd/.  To train our baseline, run the `train_model.py` in the GSDNet4Video directory.  This file takes the path to the directory containing your GSD dataset as an arugment.  Run this file like so

```
python3 train_model.py $PATH_TO_GSD_DATASET$
```

To train the three channel variant, repeat this process in the GSDNet directory.  Five different fine tuning procedures are found in FineTuningViaActiveLearning directory.  These files will all start with `fine_tune_`.  They will also take the GSD dataset's path as an argument.  These files would be run like so

```
python3 fine_tune_all_pixels.py $PATH_TO_GSD_DATASET$

python3 fine_tune_flickering_pixels.py  $PATH_TO_GSD_DATASET$

python3 fine_tune_flickering_regions.py $PATH_TO_GSD_DATASET$

python3 fine_tune_random_pixels.py $PATH_TO_GSD_DATASET$

python3 fine_tune_random_regions.py $PATH_TO_GSD_DATASET$
```

Finally, our customized version of the PixelPick GUI can be run simply with

```
python3 pixel_pick_ui.py
```
## Results

#### GSDNet vs. GSDNet4Video

Shown is a comparison of GSDNet as presented by Jiaying Lin et al. [^1] and our own modified version which uses propagation-based techniques to improves temporal stability.  Note that both these models were trained from scratch on a restricted training dataset (2710 instead of all  3202 training images) and that the ResNeXt-101 backbone has been replaced with ResNet-18.

<iframe width="560" height="315" src="https://www.youtube.com/embed/SYh0NOeJ81w" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

#### Baseline vs. Fine Tuned (trained on all pixels)

This video shows a comparison between the our baseline GSDNet4Video neural network and a fine tuned version of the same network.   The fine tuning is done using all pixels in synthetically generated videos.

<iframe width="560" height="315" src="https://www.youtube.com/embed/ZS90ZS_6w4M" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

#### Baseline vs. Fine Tuned (trained on only flickering pixels)

This video shows a comparison between the our baseline GSDNet4Video neural network and a fine tuned version of the same network.   We fine tune by training only on flickering pixels that have been detected in synthetically generated videos.

<iframe width="560" height="315" src="https://www.youtube.com/embed/UCVzW0wkBcQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>



### References

[^1]: J. Lin, Z. He, and R. W. Lau. Rich context aggregation with reflection prior for glass surface detection. - Project Page : https://jiaying.link/cvpr2021-gsd/
[^2]: V. Bazarevsky and A. Tkachenka. Mobile real-time video segmentation. - URL : https://ai.googleblog.com/2018/03/mobile-real-time-video-segmentation.html
[^3]: G. Shin, W. Xie, and S. Albanie. All you need are a few pixels: semantic segmentation with pixelpick. - GitHub : https://github.com/NoelShin/PixelPick
