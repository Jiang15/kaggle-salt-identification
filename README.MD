# TGS Salt Identification Challenge

This is a "Bronze Medal" solution to [TGS Salt Identification Challenge](https://www.kaggle.com/competitions/tgs-salt-identification-challenge)
in Kaggle. We rank 227th out of 3,726 teams (top 6%) as Bronze Medal winner.

Description: _Several areas of Earth with large accumulations of oil and gas also have huge deposits of salt below the surface.
But unfortunately, knowing where large salt deposits are precisely is very difficult. Professional seismic imaging still requires expert human interpretation of salt bodies. This leads to very subjective, highly variable renderings. More alarmingly, it leads to potentially dangerous situations for oil and gas company drillers.
To create the most accurate seismic images and 3D renderings, TGS (the world’s leading geoscience data company) is hoping Kaggle’s machine learning community will be able to build an algorithm that automatically and accurately identifies if a subsurface target is salt or not._

### Data
The data is a set of images chosen at various locations chosen at random in the subsurface. The images are 101 x 101 pixels and each pixel is classified as either salt or sediment. In addition to the seismic images, the depth of the imaged location is provided for each image. The goal of the competition is to segment regions that contain salt.
Download from [annotated_images](https://www.kaggle.com/competitions/tgs-salt-identification-challenge/data)
and place the dataset under ```input``` folder.

### Folders
- `model_zoo/`
  - `augment.py`: augment images with flipping, rotation, transform, etc.
  - `evaluate.py`: evaluate the model in terms of IOU metrics
  - `images.py`: reshape and load images for training
  - `losses.py`: implement, BCE, Lovaz losses, etc. for image segmentation
  - `inception_resnet_unet_hypercolumns_modify.py`: contains modified model architecture: unet  with inception-resnet as backbone.
- `scripts/`
    - `inception_resnet_unet_9_19_rmsprop_lovasz.py`: model training pipeline
    - `submit.ipynb`: generate prediction result for test set



### Getting Started
1. Simply run the script to train the model.
```commandline
!python scripts/inception_resnet_unet_9_19_rmsprop_lovasz.py
```


2. Run ```submit.ipynb``` to generate the result for submission.
