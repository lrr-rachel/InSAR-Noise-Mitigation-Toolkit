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

parser = argparse.ArgumentParser(description="UNet_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument('--savemodelname', type=str, default='model')
parser.add_argument('--root_distorted', type=str, default='H:/VolcanicUnrest/Atmosphere/synthesised_patches/set1/unwrap/DST_png/', help='train and test datasets')
parser.add_argument('--resultDir', type=str, default='E:/Volcano/summerproject/results_cv0/', help='save output images')
parser.add_argument('--network', type=str, default='unet')
parser.add_argument('--deform', action='store_true', help='Run test only')
parser.add_argument('--NoNorm', action='store_false', help='Run test only')
args = parser.parse_args()

num_of_layers = args.num_of_layers
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

model = UnetGenerator(input_nc=numframes, output_nc=1, num_downs=unetdepth, deformable=deform, norm_layer=NoNorm)
model.load_state_dict(torch.load(os.path.join(resultDir,'model_ep50.pth.tar'),map_location=device))
model = model.eval()
model = model.to(device)

def readimage(filename, root_distorted, numframes=1, network='unet'):
    # read distorted image
    temp = io.imread(filename,as_gray=True)
    temp = temp.astype('float32')
    image = temp/255.
    image = image[1: 225,
                      1: 225]
    image = np.expand_dims(image, axis=2)
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image)
    vallist = [0.5]*image.shape[0]
    normmid = transforms.Normalize(vallist, vallist)
    image = normmid(image)
    image = image.unsqueeze(0)
    return image
# =====================================================================

numframes = 1
filesnames = glob.glob(os.path.join(root_distorted,'*.png'))
for i in range(len(filesnames[:10])):
    curfile = filesnames[i]
    inputs = readimage(curfile, root_distorted, numframes, network=network)
    subname = curfile.split("\\")
    inputs = inputs.to(device)
    with torch.no_grad():
        output = model(inputs)
        output = output.squeeze(0)
        output = output.cpu().numpy() 
        output = output.transpose((1, 2, 0))
        output = (output*0.5 + 0.5)*255
        io.imsave(os.path.join(resultDirOutImg, subname[-1]), output.astype(np.uint8))
        print("Image Saved...")