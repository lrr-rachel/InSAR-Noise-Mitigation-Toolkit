import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import data_augmentation
from sklearn.model_selection import KFold
import random


def split_trainval(noisy_path):
    '''
    Simply splitting data to train and val in 4:1 (if harcode foldNum = 0)
    for 10,000 data: each fold train with 8000, val with 2000 data
    '''
    # 5-fold cross validation
    folds = KFold(n_splits=5, shuffle=True, random_state=1)
    noisy_list = os.listdir(noisy_path)
    # same index for both noisy and clean data
    trainindx_N = []
    valindx_N = []
    # manually set the fold number  
    foldNum = 0
    for fold_i, (train_index, val_index) in enumerate(folds.split(noisy_list)):
        if fold_i == foldNum: 
            trainindx_N  = train_index
            valindx_N  = val_index

    return trainindx_N, valindx_N

def normalize(data):
    return data/255.

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def prepare_data(data_path, noisy, patch_size, stride):
    trainindx, valindx = split_trainval(data_path)
    files = glob.glob(os.path.join(data_path, '*.png'))
    files.sort()
    if noisy == False:
        print('process CLEAN training data')
        h5f = h5py.File('train.h5', 'w')
    else: 
        print('process NOISY training data')
        h5f = h5py.File('train_noisy.h5', 'w')
    train_num = 0
    for i in range(len(trainindx)):
        img = cv2.imread(files[trainindx[i]])
        h, w, c = img.shape
        Img = cv2.resize(img, (int(h*1), int(w*1)), interpolation=cv2.INTER_CUBIC)
        Img = np.expand_dims(Img[:,:,0].copy(), 0)
        Img = np.float32(normalize(Img))
        patches = Im2Patch(Img, win=patch_size, stride=stride)
        print("file: %s scale %.1f # samples: %d" % (files[trainindx[i]], 1, patches.shape[3]))
        for n in range(patches.shape[3]):
            data = patches[:,:,:,n].copy()
            h5f.create_dataset(str(train_num), data=data)
            train_num += 1

    h5f.close()
    # val
    files.clear()
    files = glob.glob(os.path.join(data_path, '*.png'))
    files.sort()
    if noisy == False:
        print('\nprocess CLEAN validation data')
        h5f = h5py.File('val.h5', 'w')
    else:
        print('\nprocess NOISY validation data')
        h5f = h5py.File('val_noisy.h5', 'w')
    val_num = 0
    for i in range(len(valindx)):
        print("file: %s" % files[valindx[i]])
        img = cv2.imread(files[valindx[i]])
        img = np.expand_dims(img[:,:,0], 0)
        img = np.float32(normalize(img))
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()
    print('training set, # samples %d\n' % train_num)
    print('val set, # samples %d\n' % val_num)

class Dataset(udata.Dataset):
    def __init__(self, train=True):
        super(Dataset, self).__init__()
        self.train = train
        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()

        if self.train:
            h5f_noisy = h5py.File('train_noisy.h5', 'r')
        else:
            h5f_noisy = h5py.File('val_noisy.h5', 'r')
        self.noisykeys = list(h5f_noisy.keys())
        random.shuffle(self.noisykeys)
        h5f_noisy.close()
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):
        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        if self.train:
            h5f_noisy = h5py.File('train_noisy.h5', 'r')
        else:
            h5f_noisy = h5py.File('val_noisy.h5', 'r')
        noisykey = self.keys[index]
        noisydata = np.array(h5f_noisy[noisykey])
        h5f.close()
        return (torch.Tensor(data), torch.Tensor(noisydata))



