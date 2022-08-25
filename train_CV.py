from __future__ import print_function, division

import argparse
from ctypes import resize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy
import glob
from torch.utils.data import Dataset
from skimage import io, transform
from networks import UnetGenerator
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import random


parser = argparse.ArgumentParser(description='Main Unet')
parser.add_argument('--root_distorted', type=str, default='E:/Volcano/summerproject/synthetic_datasets/DST_png/', help='train and test datasets')
parser.add_argument('--root_restored', type=str, default='E:/Volcano/summerproject/synthetic_datasets/D_png/', help='save output images')
parser.add_argument('--resultDir', type=str, default='results_cv0_epoch10', help='save output images')
parser.add_argument('--unetdepth', type=int, default=5, metavar='N',  help='number of epochs to train (default: 10)')
parser.add_argument('--numframes', type=int, default=1, metavar='N',  help='number of epochs to train (default: 10)')
parser.add_argument('--cropsize', type=int, default=224)
parser.add_argument('--savemodelname', type=str, default='model')
parser.add_argument('--maxepoch', type=int, default=100)
parser.add_argument('--NoNorm', action='store_false', help='Run test only')
parser.add_argument('--deform', action='store_true', help='Run test only')
parser.add_argument('--network', type=str, default='unet')
parser.add_argument('--retrain', action='store_true')
parser.add_argument('--topleft', action='store_true', help='crop using top left')
parser.add_argument('--resizedata',default='false', help='resize input')
parser.add_argument('--resize_height',type=int,default=256,help='resize height')
parser.add_argument('--resize_width',type=int,default=256, help='resize width')
parser.add_argument('--foldNum', type=int, default=0, help='cross validation fold number: 0-4')

args = parser.parse_args()

root_distorted = args.root_distorted
root_restored  = args.root_restored
resultDir = args.resultDir
unetdepth = args.unetdepth
numframes = args.numframes
cropsize = args.cropsize
savemodelname = args.savemodelname
maxepoch = args.maxepoch
NoNorm = True# args.NoNorm
deform = args.deform
network = args.network
retrain = args.retrain
topleft = args.topleft
resizedata = args.resizedata
resize_height = args.resize_height
resize_width = args.resize_width
foldNum = args.foldNum

# root_distorted='datasets/van_distorted'
# root_restored='datasets/van_restored'
# resultDir = 'results'
if not os.path.exists(resultDir):
    os.mkdir(resultDir)

class Dataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, root_distorted, root_restored='', network='unet', numframes=1, transform=None):
        self.root_distorted = root_distorted
        self.root_restored = root_restored
        self.transform = transform
        if len(root_restored)==0:
            self.filesnames = glob.glob(os.path.join(root_distorted,'**_restored/*.png'))
        else:
            self.filesnames = glob.glob(os.path.join(root_distorted,'*.png'))
        self.numframes = numframes
    def __len__(self):
        return len(self.filesnames)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        subname = self.filesnames[idx].split("\\")
        totalframes = len(self.filesnames)
        # read distorted image
        temp = io.imread(os.path.join(self.root_distorted,subname[-1]),as_gray=True)
        temp = temp.astype('float32')
        #if f==rangef[0]:
        image = temp/255.
        # read corresponding clean image
        # print("restored:",self.root_restored)
        groundtruth = io.imread(os.path.join(self.root_restored,subname[-1]),as_gray=True)
        groundtruth = groundtruth.astype('float32')
        groundtruth = groundtruth/255.
        sample = {'image': image, 'groundtruth': groundtruth}
        if self.transform:
            sample = self.transform(sample)
        return sample


class RandomCrop(object):
    def __init__(self, output_size, topleft=False):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.topleft = topleft
    def __call__(self, sample):
        image, groundtruth = sample['image'], sample['groundtruth']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        if (h > new_h) and (not topleft):
            top = np.random.randint(0, h - new_h)
        else:
            top = 0
        if (w > new_w) and (not topleft):
            left = np.random.randint(0, w - new_w)
        else:
            left = 0
        image = image[top: top + new_h,
                      left: left + new_w]
        groundtruth = groundtruth[top: top + new_h,
                      left: left + new_w]
        return {'image': image, 'groundtruth': groundtruth}

class ToTensor(object):
    def __init__(self, network='unet'):
        self.network = network
    def __call__(self, sample):
        image, groundtruth = sample['image'], sample['groundtruth']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = np.expand_dims(image, axis=2)
        groundtruth = np.expand_dims(groundtruth, axis=2)
        image = image.transpose((2, 0, 1))
        groundtruth = groundtruth.transpose((2, 0, 1))
        image = torch.from_numpy(image.copy())
        groundtruth = torch.from_numpy(groundtruth.copy())
        # image
        vallist = [0.5]*image.shape[0]
        normmid = transforms.Normalize(vallist, vallist)
        image = normmid(image)
        # ground truth
        vallist = [0.5]*groundtruth.shape[0]
        normmid = transforms.Normalize(vallist, vallist)
        groundtruth = normmid(groundtruth)
        image = image.unsqueeze(0)
        groundtruth = groundtruth.unsqueeze(0)
        return {'image': image, 'groundtruth': groundtruth}

class RandomFlip(object):
    def __call__(self, sample):
        image, groundtruth = sample['image'], sample['groundtruth']
        op = np.random.randint(0, 3)
        if op<2:
            image = np.flip(image,op)
            groundtruth = np.flip(groundtruth,op)
        return {'image': image, 'groundtruth': groundtruth}

def readimage(filename, root_distorted, numframes=1, network='unet'):
    # read distorted image
    temp = io.imread(filename,as_gray=True)
    temp = temp.astype('float32')
    temp = temp[1: 225, 1: 225]
    image = temp/255.
    image = np.expand_dims(image, axis=2)
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image)
    vallist = [0.5]*image.shape[0]
    normmid = transforms.Normalize(vallist, vallist)
    image = normmid(image)
    image = image.unsqueeze(0)
    return image

def resizeimage(path, width, height):
    print("[INFO] resizing inputs to ", resize_width, 'x', resize_height)
    dirs_dis = os.listdir(path)
    for item in dirs_dis:
        if os.path.isfile(path+'/'+item):
            im = Image.open(path+'/'+item)
            f, e = os.path.splitext(path+'/'+item)
            imResize = im.resize((width,height), Image.ANTIALIAS)
            imResize.save(f + '.png', 'PNG', quality=90)
            # print('image resize')
# =====================================================================

# resize input image size
if resizedata == 'true':
    print('[INFO] resize noisy input...')
    resizeimage(root_distorted,resize_width,resize_height)
    print('[INFO] resize clean input...')
    resizeimage(root_restored,resize_width,resize_height)

# data loader
print("[INFO] Loading Data")
if cropsize==0:
    dataset = Dataset(root_distorted=root_distorted,
                                    root_restored=root_restored, network=network, numframes=numframes,
                                    transform=transforms.Compose([RandomFlip(),ToTensor(network=network)]))
else:
    dataset = Dataset(root_distorted=root_distorted,
                                    root_restored=root_restored, network=network, numframes=numframes,
                                    transform=transforms.Compose([RandomCrop(cropsize, topleft=topleft),RandomFlip(),ToTensor(network=network)]))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("[INFO] Generating UNet")
model = UnetGenerator(input_nc=numframes, output_nc=1, num_downs=unetdepth, deformable=deform, norm_layer=NoNorm)
if retrain:
    model.load_state_dict(torch.load(os.path.join(resultDir,savemodelname+'.pth.tar'),map_location=device))

model = model.to(device)

criterion = nn.MSELoss()

# Observe that all parameters are being optimized
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 5-fold cross validation 
folds = KFold(n_splits=5, shuffle=True, random_state=1)
image_path_list = os.listdir(root_distorted)
label_list = os.listdir(root_restored)

train_data = []
val_data = []
# manually set the fold number 
# each fold train with 8000, val with 2000
for fold_i, (train_index, val_index) in enumerate(folds.split(image_path_list)):
    if fold_i == foldNum: 
        train_data = train_index
        val_data = val_index

# print('train data:',train_data)
# print('val data:',val_data)
# print('train length:',len(train_data))
# print('val length:',len(val_data))
# =====================================================================
print('---------------- Fold ', foldNum, ' ----------------')
print("[INFO] Training...")
num_epochs=maxepoch
since = time.time()
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 100000000.0

# plot loss curve
train_graph = []
val_graph = []

for epoch in range(num_epochs+1):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    # Each epoch has a training and validation phase
    # for phase in ['train']:#, 'val']:
    for phase in ['train','val']:
        running_loss = 0.0
        running_corrects = 0
        validate_loss = 0.0
        validate_corrects = 0
        if phase == 'train':
            # print('[INFO] Training Phase...')
            model = model.train()  # Set model to training mode
            # Iterate over train data.
            for i in random.sample(range(len(train_data)),len(train_data)):
                sample = dataset[train_data[i]]
                inputs = sample['image'].to(device)
                labels = sample['groundtruth'].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                # with torch.no_grad():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    # loss.requires_grad_(True)
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                # print('running loss:',running_loss)
            epoch_loss = running_loss / len(train_data)
            train_graph.append(epoch_loss)
            # print('\n')
            print('[Epoch] ' + str(epoch),':' + '[Train Loss] ' + str(epoch_loss))
            # print('\n')
        if phase == 'val':
            # print('[INFO] Evaluate Phase...')
            model = model.eval()   # Set model to evaluate mode
            for i in range(len(val_data)):
                sample_val = dataset[val_data[i]]
                inputs_val = sample_val['image'].to(device)
                labels_val = sample_val['groundtruth'].to(device)
                outputs_val = model(inputs_val)
                loss_val = criterion(outputs_val, labels_val)
                # statistics
                validate_loss += loss_val.item() * inputs_val.size(0)
            val_loss = validate_loss / len(val_data)
            val_graph.append(val_loss)
            print('[Epoch] ' + str(epoch),':' + '[Val Loss] ' + str(val_loss))
        # save output_val to 10 images to see when it's wrapped and compare with gt
        if (epoch % 10) == 0:
            torch.save(model.state_dict(), os.path.join(resultDir, savemodelname + '_ep'+str(epoch)+'.pth.tar'))
            filesnames = glob.glob(os.path.join(root_distorted,'*.png'))
            save_dir = os.path.join(resultDir,'epoch '+str(epoch))
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            #save outputs (training outputs)
            resultDirTrain = os.path.join(save_dir,savemodelname+'_train')
            if not os.path.exists(resultDirTrain):
                os.mkdir(resultDirTrain)
            model = model.to(device)
            with torch.no_grad():
                for i in range(len(train_data[:10])):
                    curfile = filesnames[train_data[i]]
                    inputs = readimage(curfile, root_distorted, numframes, network=network)
                    subname = curfile.split("\\")
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    outputs = outputs.squeeze(0)
                    outputs = outputs.cpu().numpy() 
                    outputs = outputs.transpose((1, 2, 0))
                    outputs = (outputs*0.5 + 0.5)*255
                    io.imsave(os.path.join(resultDirTrain, subname[-1]), outputs.astype(np.uint8))
            # save outputs_val 
            resultDirVal = os.path.join(save_dir,savemodelname+'_val')
            if not os.path.exists(resultDirVal):
                os.mkdir(resultDirVal)
            model = model.eval()
            model = model.to(device)
            with torch.no_grad():
                for i in range(len(val_data[:10])):
                    curfile = filesnames[val_data[i]]
                    inputs_val = readimage(curfile, root_distorted, numframes, network=network)
                    subname = curfile.split("\\")
                    inputs_val = inputs_val.to(device)
                    outputs_val = model(inputs_val)
                    outputs_val = outputs_val.squeeze(0)
                    outputs_val = outputs_val.cpu().numpy() 
                    outputs_val = outputs_val.transpose((1, 2, 0))
                    outputs_val = (outputs_val*0.5 + 0.5)*255
                    io.imsave(os.path.join(resultDirVal, subname[-1]), outputs_val.astype(np.uint8))
        # deep copy the model
        if (epoch>10) and (epoch_loss < best_acc):
            best_acc = epoch_loss
            torch.save(model.state_dict(), os.path.join(resultDir, 'best_'+savemodelname+'.pth.tar'))

# plot loss graph
plt.figure(figsize=(10,5))
plt.title('Training and Validate Loss')
plt.plot(train_graph,label="training loss")
plt.plot(val_graph,label="validate loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
figname = 'Loss Function '+ 'Fold'+ str(foldNum) +'.jpg'
#plt.savefig(figname)
plt.show()

# # =======TESTING==============================================================
# resultDirOutImg = os.path.join(resultDir,savemodelname)
# if not os.path.exists(resultDirOutImg):
#     os.mkdir(resultDirOutImg)

# model = UnetGenerator(input_nc=numframes, output_nc=1, num_downs=unetdepth, deformable=deform, norm_layer=NoNorm)
# model.load_state_dict(torch.load(os.path.join(resultDir,'best_'+savemodelname+'.pth.tar'),map_location=device))
# model.eval()
# model = model.to(device)

# # =====================================================================
# filesnames = glob.glob(os.path.join(root_distorted,'*.png'))
# for i in range(len(filesnames[:10])):
#      curfile = filesnames[i]
#      inputs = readimage(curfile, root_distorted, numframes, network=network)
#      subname = curfile.split("\\")
#      inputs = inputs.to(device)
#      with torch.no_grad():
#          output = model(inputs)
#          output = output.squeeze(0)
#          output = output.cpu().numpy() 
#          output = output.transpose((1, 2, 0))
#          output = (output*0.5 + 0.5)*255
#          io.imsave(os.path.join(resultDirOutImg, subname[-1]), output.astype(np.uint8))




