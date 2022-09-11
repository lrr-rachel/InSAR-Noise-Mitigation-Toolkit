# About the Project

Interferometric Synthetic Aperture Radar (InSAR) is a technique that can be used to make high-density measurements of spatial extent and magnitude of surface deformation over large areas.

With the help of InSAR technique, we can detect phenomena such as volcanic eruption or earthquake, and perform prediction. However, there are interferences such as atmosphere artifacts, or water vapor that could affect the detection. Since these noisy effects can look like fringes in interferograms, it is hard to distinguish between the noisy effects and the real deformation signals. Therefore, this project introduces a mitigation technique to remove the noisy atmosphere artifacts, and more importantly, improve the detection accuracy. 

* This is a Pytorch implementation for mitigating the noisy effect on InSAR images and improving the detection accuracy. Two different models (UNet & DnCNN) can be used in the mitigation process. Please only select one model you want to use in Training and Testing. 

## Models

### UNet

The UNet implementation is based on the structure proposed in [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).

U-Net has been a widely used technique in many visual tasks, especially in the fields where localization, or pixel-level prediction, is required in the expected output. U-Net’s convolutional layers are used to conduct semantic segmentation tasks.

U-Net consists of two parts, Encoder and Decoder. In the Encoder, the size of the image gradually reduces (down-samples) while the depth gradually increases. But encoder is not able to preserve precise localization. Decoder up-samples the image from low-resolution to high resolution, but also restores information about where the object is (preserves localization).

### DnCNN

The DnCNN (Denoising Convolutional Neural Network) implementation is based on the structure proposed in [Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](https://ieeexplore.ieee.org/document/7839189) and modified from [DnCNN-PyTorch](https://github.com/SaoYan/DnCNN-PyTorch)

Denoising Convolutional Neural Network (DnCNN) is another well-known architecture for denoising. Batch normalization and residual learning are used to further improve the performance (Notice that the noise in InSAR is not gaussian distributed).

Unlike UNet, DnCNN does not directly output the clean image, the model is trained to distinguish the difference between the noisy image and the clean ground truth. With the residual learning, DnCNN can implicitly remove the latent clean image in the hidden layers, and then output the additional noise. Noise is then subtracted from the noisy image hence outputting a clean image. Thus, DnCNN can gradually separate image structure from the noisy observation.

Note: in this project running ```test.py``` for DnCNN will directly output clean images for convenience.

## Environment
* BlueCrystal4: Recommand building your own anaconda environment. Please see detailed instrusctions on [installing your own conda env on bc4](https://www.acrc.bris.ac.uk/protected/hpc-docs/software/python_conda.html)

## Getting Started
* Two models - UNet and DnCNN are avialable for usage.
* Make sure the input images are unwrapped.
* Remember to specify the model network you want to use in training (```train.py```) and testing (```test.py```) file via command line arguements. The default model is UNet.

### Training
* Run ```train.py```
* Specify the model network (unet/dncnn), noisy data directory, clean data directory, output directory etc,.
* run ```train.py --help```  to see useful help messages.
* Trained model result will be saved in the output directory.

Note: Please see help messages for command line arguements before continuing.

#### Quick UNet Example:
Output a trained UNet model named 'best_unetmodel.pth.tar' in output directory 'unetresults'

```
python train.py --network unet --root_distorted ./datasets/DST_png/ --root_restored ./datasets/D_png/ --resultDir unetresults --savemodelname unetmodel --maxepoch 200 
```

#### Quick DnCNN Example:
Output a trained DnCNN model named 'best_dncnnmodel.pth.tar' in output directory 'dncnnresults'

If you've already built the training and validation dataset (i.e. ```train.h5, val.h5 files, train_noisy.h5, val_noisy.h5``` files), set ```preprocess``` to be ```False```. Otherwise, if training DnCNN for the first time, set ```preprocess``` to be ```True```.

```
python train.py --network dncnn --preprocess True --root_distorted ./datasets/DST_png/ --root_restored ./datasets/D_png/ --resultDir dncnnresults --savemodelname dncnnmodel --maxepoch 200 
```

### Testing
* Run ```test.py``` 
* Specify the model network (unet/dncnn), test data directory, output result directory etc,.
* run ```test.py --help```  to see useful help messages.
* ```test.py``` loads and tests the trained model using noisy test dataset, then outputs corresponding clean images (unwrapped & wrapped).

Note: 
* Please see help messages for command line arguements before continuing.
* The ```resultDir``` must be the output direcory created in the Training stage. 
* ```resultDir``` must contain the trained model.


#### Quick UNet Example:
```
python test.py --network unet --root_distorted ./test_dataset/ --resultDir unetresults
```

#### Quick DnCNN Example:
```
python test.py --network dncnn --root_distorted ./test_dataset/ --resultDir dncnnresults
```