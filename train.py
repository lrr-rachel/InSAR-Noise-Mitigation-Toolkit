from __future__ import print_function, division
import argparse
from ctypes import resize
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import  transforms
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
from models import DnCNN
from dataset import *
from utils import *
from torch.autograd import Variable
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Main Training Code')
parser.add_argument('--network', type=str, default='unet',help='select model: unet or dncnn')
parser.add_argument('--root_distorted', type=str, default='/user/work/datasets/DST_png/', help='noisy dataset')
parser.add_argument('--root_restored', type=str, default='/user/work/datasets/D_png/', help='clean dataset')
parser.add_argument('--resultDir', type=str, default='results', help='output directory')
parser.add_argument('--unetdepth', type=int, default=5, metavar='N',  help='number of unet depths (default: 5)')
parser.add_argument('--numframes', type=int, default=32, metavar='N',  help='batch number (default: 32)')
parser.add_argument('--maxepoch', type=int, default=200, help='number of epochs to train. default: 200')
parser.add_argument('--savemodel_epoch', type=int, default=20, help='save model every _ epochs. default: 20 (save model every 20 epochs)')
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--preprocess", type=bool, default=False, help='prepare DnCNN data or not')
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers in DnCNN")
parser.add_argument("--batchnorm", type=bool, default=False, help='use batch normalization in DnCNN')
parser.add_argument('--cropsize', type=int, default=224)
parser.add_argument('--savemodelname', type=str, default='model')
parser.add_argument('--NoNorm', action='store_false', help='Run test only')
parser.add_argument('--deform', action='store_true', help='Run test only')
parser.add_argument('--retrain', action='store_true', help='Retrain')
parser.add_argument('--topleft', action='store_true', help='crop using top left')
parser.add_argument('--resizedata',default='false', help='resize input')
parser.add_argument('--resize_height',type=int,default=256,help='resize height')
parser.add_argument('--resize_width',type=int,default=256, help='resize width')
# parser.add_argument('--foldNum', type=int, default=0, help='manually set 5-f cross validation fold number: 0-4')



args = parser.parse_args()

network = args.network
root_distorted = args.root_distorted
root_restored  = args.root_restored
resultDir = args.resultDir
unetdepth = args.unetdepth
numframes = args.numframes
maxepoch = args.maxepoch
savemodel_epoch = args.savemodel_epoch
lr = args.lr
preprocess = args.preprocess
num_of_layers = args.num_of_layers
batchnorm = args.batchnorm
cropsize = args.cropsize
savemodelname = args.savemodelname
NoNorm = args.NoNorm
deform = args.deform
retrain = args.retrain
topleft = args.topleft
resizedata = args.resizedata
resize_height = args.resize_height
resize_width = args.resize_width
# foldNum = args.foldNum


if not os.path.exists(resultDir):
    os.mkdir(resultDir)

class UNetDataset(Dataset):
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
        #curf = int(subname[-1][:-4])
        #halfcurf = int(self.numframes/2)
        if len(self.root_restored)==0:
            totalframes = len(glob.glob(os.path.join(os.path.dirname(os.path.abspath(self.filesnames[idx])), '*.png')))
        else:
            totalframes = len(self.filesnames)
        if numframes > 1:
            otherindx = random.sample(range(totalframes),numframes-1)
            rangef = np.unique(np.append(idx, otherindx))
            while len(rangef) < numframes:
                rangef = np.unique(np.append(rangef, random.sample(range(totalframes),numframes-len(rangef))))
        for f in rangef:
            subname = self.filesnames[f].split("\\")
            # read distorted image
            temp = io.imread(os.path.join(self.root_distorted,subname[-1]),as_gray=True)
            temp = temp.astype('float32')
            temp = temp[..., np.newaxis]
            # read restored image
            tempgt = io.imread(os.path.join(self.root_restored,subname[-1]),as_gray=True)
            tempgt = tempgt.astype('float32')
            tempgt = tempgt[..., np.newaxis]
            if f==rangef[0]:
                image = temp/255.
                groundtruth = tempgt/255.
            else:
                image = np.append(image,temp/255.,axis=2)
                groundtruth = np.append(groundtruth,tempgt/255.,axis=2)
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
        # numpy image: H x W x B
        # torch image: B x H x W
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
        # torch image: B x 1 x H x W
        image = image.unsqueeze(1)
        groundtruth = groundtruth.unsqueeze(1)
        return {'image': image, 'groundtruth': groundtruth}

class RandomFlip(object):
    def __call__(self, sample):
        image, groundtruth = sample['image'], sample['groundtruth']
        op = np.random.randint(0, 3)
        if op<2:
            image = np.flip(image,op)
            groundtruth = np.flip(groundtruth,op)
        return {'image': image, 'groundtruth': groundtruth}

def readimage(filename):
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# data loader
print("[INFO] Loading Data")
if network == 'unet':
    if cropsize==0:
        unetdataset = UNetDataset(root_distorted=root_distorted,
                                        root_restored=root_restored, network=network, numframes=numframes,
                                        transform=transforms.Compose([RandomFlip(),ToTensor(network=network)]))
    else:
        unetdataset = UNetDataset(root_distorted=root_distorted,
                                        root_restored=root_restored, network=network, numframes=numframes,
                                        transform=transforms.Compose([RandomCrop(cropsize, topleft=topleft),RandomFlip(),ToTensor(network=network)]))

    print("[INFO] Generating UNet")
    model = UnetGenerator(input_nc=1, output_nc=1, num_downs=unetdepth, deformable=deform, norm_layer=NoNorm)
    if retrain:
        model.load_state_dict(torch.load(os.path.join(resultDir,'best_'+ savemodelname+'.pth.tar'),map_location=device))
    model = model.to(device)
    criterion = nn.MSELoss()
else:
    if preprocess:
        # prepare clean dataset
        prepare_data(data_path=root_restored, noisy=False, patch_size=50, stride=10)
        # prepare noisy dataset
        prepare_data(data_path=root_distorted, noisy=True, patch_size=50, stride=10)
    train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=train, num_workers=4, batch_size=numframes, shuffle=True)
    # Build model
    print("[INFO] Generating DnCNN")
    net = DnCNN(channels=1, num_of_layers=num_of_layers, bn=False)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False)
    model = nn.DataParallel(net, device_ids=[0]).cuda()
    if retrain:
        model.load_state_dict(torch.load(os.path.join(resultDir,'best_'+ savemodelname+'.pth.tar'),map_location=device))
    criterion.cuda()

# Observe that all parameters are being optimized
optimizer = optim.Adam(model.parameters(), lr=lr)

# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

train_data, val_data = split_trainval(root_distorted)

# =====================================================================
print("[INFO] Training...")
num_epochs=maxepoch
since = time.time()
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 100000000.0

# plot loss and val curve
train_graph = []
val_graph = []

for epoch in range(num_epochs+1):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    dncnn_loss = 0.0
    # Each epoch has a training and validation phase
    for phase in ['train','val']:
        running_loss = 0.0
        running_corrects = 0
        validate_loss = 0.0
        validate_corrects = 0
        if phase == 'train':
            model = model.train()  # Set model to training mode
            # Iterate over train data.
            if network == 'unet':
                for i in random.sample(range(len(train_data)),len(train_data)):
                    sample = unetdataset[train_data[i]]
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
                    if (i % 1000) == 0:
                        print(str(i) + ' running loss:' + str(running_loss))
                epoch_loss = running_loss / len(train_data) / numframes
            else:
                for i, (data,noisy) in enumerate(loader_train, 0):
                    # training dncnn
                    model.train()
                    model.zero_grad()
                    optimizer.zero_grad()
                    img_train = data #clean
                    img_noisy = noisy #noisy
                    imgn_train = img_noisy
                    with torch.no_grad():
                        img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
                        out_train = model(imgn_train)
                        loss = criterion(out_train, imgn_train-img_train) / (imgn_train.size()[0]*2)
                        loss.requires_grad_(True)
                        loss.backward()
                        optimizer.step()
                    dncnn_loss += loss.item()
                    
                epoch_loss = dncnn_loss/len(loader_train)/numframes
            train_graph.append(epoch_loss)
            print('[Epoch] ' + str(epoch),':' + '[Train Loss] ' + str(epoch_loss))

        if phase == 'val':
            model = model.eval()   # Set model to evaluate mode
            if network == 'unet':
                for i in range(len(val_data)):
                    sample_val = unetdataset[val_data[i]]
                    inputs_val = sample_val['image'].to(device)
                    labels_val = sample_val['groundtruth'].to(device)
                    outputs_val = model(inputs_val)
                    loss_val = criterion(outputs_val, labels_val)
                    # statistics
                    validate_loss += loss_val.item() * inputs_val.size(0)
                val_loss = validate_loss / len(val_data)
                val_graph.append(val_loss)
                print('[Epoch] ' + str(epoch),':' + '[Val Loss] ' + str(val_loss))
            else: 
                # validate
                for k, (data,noisy) in enumerate(dataset_val, 0):
                    img_val = torch.unsqueeze(data, 0)
                    img_valnoisy = torch.unsqueeze(noisy,0)
                    imgn_val = img_valnoisy
                    with torch.no_grad():
                        img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda())
                        out_val = torch.clamp(imgn_val-model(imgn_val), 0., 1.)
                        loss_val = criterion(out_val, imgn_val-img_val) / (imgn_val.size()[0]*2)
                        loss_val.requires_grad_(True)
        # save model
        if (epoch % savemodel_epoch) == 0:
            torch.save(model.state_dict(), os.path.join(resultDir, savemodelname + '_ep'+str(epoch)+'.pth.tar'))

        # deep copy the model
        if network == 'unet':
            if (epoch>1) and (epoch_loss < best_acc):
                best_acc = epoch_loss
                torch.save(model.state_dict(), os.path.join(resultDir, 'best_'+savemodelname+'.pth.tar'))
        else:
            torch.save(model.state_dict(), os.path.join(resultDir, 'best_'+savemodelname+'.pth.tar'))

# plot loss and val graph
plt.figure(figsize=(10,5))
plt.title('Training and Validate Loss')
plt.plot(train_graph,label="training loss")
plt.plot(val_graph,label="validate loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
figname = os.path.join(resultDir, network + ' Loss and Val Graph.jpg')
plt.savefig(figname)
#plt.show()




