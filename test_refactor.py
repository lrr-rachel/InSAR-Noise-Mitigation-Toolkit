import argparse
import torch
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from skimage import io, transform
from torchvision import transforms
from networks import *
import cv2

parser = argparse.ArgumentParser(description="Test")
parser.add_argument('--network', type=str, default='unet',help='select model: unet or dncnn')
parser.add_argument('--savemodelname', type=str, default='model')
parser.add_argument('--root_distorted', type=str, default='DST_png', help='noisy dataset')
parser.add_argument('--resultDir', type=str, default='results', help='save output images. default: results (same dir as resultDir in train.py)')
parser.add_argument('--deform', action='store_true', help='Run test only')
parser.add_argument('--NoNorm', action='store_false', help='Run test only')
args = parser.parse_args()

network = args.network
savemodelname = args.savemodelname
root_distorted = args.root_distorted
resultDir = args.resultDir
network = args.network
deform = args.deform
NoNorm = args.NoNorm

# =======TESTING==============================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
numframes = 1
unetdepth = 5

resultDirOutImg = os.path.join(resultDir,savemodelname)
if not os.path.exists(resultDirOutImg):
    os.mkdir(resultDirOutImg)

unwrappedDir= os.path.join(resultDirOutImg,'unwrapped')
if not os.path.exists(unwrappedDir):
    os.mkdir(unwrappedDir)

wrappedDir = os.path.join(resultDirOutImg,'wrapped')
if not os.path.exists(wrappedDir):
    os.mkdir(wrappedDir)


if network == 'unet':
    model = UnetGenerator(input_nc=numframes, output_nc=1, num_downs=unetdepth, deformable=deform, norm_layer=NoNorm)
    # load saved model in restultDir
    model.load_state_dict(torch.load(os.path.join(resultDir,'best_model.pth.tar'),map_location=device))

model = model.eval()
model = model.to(device)

def readimage(filename):
    # read distorted image
    temp = io.imread(filename,as_gray=True)
    temp = temp.astype('float32')
    image = temp/255.
    # image = image[1: 225,1: 225]
    image = np.expand_dims(image, axis=2)
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image)
    vallist = [0.5]*image.shape[0]
    normmid = transforms.Normalize(vallist, vallist)
    image = normmid(image)
    image = image.unsqueeze(0)
    return image

def im2double(im):
  np.seterr(divide='ignore', invalid='ignore')
  min_val = np.min(im.ravel())
  max_val = np.max(im.ravel())
  out = (im.astype('float') - min_val) / (max_val - min_val)
  return out
# =====================================================================

# save unwrapped output images
filesnames = glob.glob(os.path.join(root_distorted,'*.png'))
for i in range(len(filesnames[:10])):
    curfile = filesnames[i]
    inputs = readimage(curfile)
    subname = curfile.split("\\")
    inputs = inputs.to(device)
    with torch.no_grad():
        output = model(inputs)
        output = output.squeeze(0)
        output = output.cpu().numpy() 
        output = output.transpose((1, 2, 0))
        output = (output*0.5 + 0.5)*255
        io.imsave(os.path.join(unwrappedDir, subname[-1]), output.astype(np.uint8))
        #print("Unwrapped Image Saved")

# save wrapped output images
dirs = os.listdir(root_distorted)
for item in dirs[:10]:
    fname = item.split("\\")
    I = cv2.imread(os.path.join(root_distorted,item),cv2.IMREAD_GRAYSCALE) 
    print(I)
    I = im2double(I)
    # convert grayscale to rad interferogram
    radI = I*100 - 30
    # wrapping
    wrappedI = radI % (2 * np.pi)
    # rescale
    grayI = wrappedI/(2*np.pi) 
    rgbI = cv2.applyColorMap((grayI*255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(wrappedDir, fname[-1]), rgbI)
    # print('image saved')