# About the Project

This is a Pytorch implementation for mitigating the noisy effect on InSAR images and improving the detection accuracy.

## Models

### UNet

The UNet implementation is based on the structure proposed in [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).

U-Net has been a widely used technique in many visual tasks, especially in the fields where localization, or pixel-level prediction, is required in the expected output. U-Net’s convolutional layers are used to conduct semantic segmentation tasks.

U-Net consists of two parts, Encoder and Decoder. In the Encoder, the size of the image gradually reduces (down-samples) while the depth gradually increases. But encoder is not able to preserve precise localization. Decoder up-samples the image from low-resolution to high resolution, but also restores information about where the object is (preserves localization).

### DnCNN

The DnCNN (Denoising Convolutional Neural Network) implementation is based on the structure proposed in [Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](https://ieeexplore.ieee.org/document/7839189) and modified from [DnCNN-PyTorch](https://github.com/SaoYan/DnCNN-PyTorch)

Denoising Convolutional Neural Network (DnCNN) is another well-known architecture for denoising. Batch normalization and residual learning are used to further improve the performance (Notice that the noise in InSAR is not gaussian distributed).

Unlike UNet, DnCNN does not directly output the clean image, the model is trained to distinguish the difference between the noisy image and the clean ground truth. With the residual learning, DnCNN can implicitly remove the latent clean image in the hidden layers, and then output the additional noise. Noise is then subtracted from the noisy image hence outputting a clean image. Thus, DnCNN can gradually separate image structure from the noisy observation.

Note: in this project running 'test.py' for DnCNN will directly output clean images for convenience.

## Environment
* BlueCrystal4: Recommand building your own anaconda environment. Please see detailed instrusctions on [installing your own conda env on bc4](https://www.acrc.bris.ac.uk/protected/hpc-docs/software/python_conda.html)

## Getting Started
* Two models - UNet and DnCNN are avialable for usage.
* Remember to specify the model you want to use in training (```train.py```) and testing (```test.py```) file via command line arguements.

### Training
* Run ```train.py```
* Specify the model network, input noisy data directory, input clean data directory, output result directory etc,.
* run ```train.py --help```  to see useful help messages.

quick example:
```
python train.py 
  --network unet 
  --root_distorted ./datasets/DST_png/ 
  --root_restored ./datasets/D_png/ 
  --resultDir results
  --maxepoch 200
```
* TODO: fix DnCNN 

### Testing
* Run ```test.py``` 
* Specify the model network, input test data directory, output result directory etc,.
* run ```test.py --help```  to see useful help messages.

* TODO: add detailed explanation

quick example:
```
python test.py 
  --network unet 
  --root_distorted ./datasets/DST_png/ 
  --resultDir results
```