import os
import numpy as np
from tqdm import tqdm
import scipy.io as so
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sleepy
from utils import load_yaml, new_remidx
from multiprocessing import Process
import shutil

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torchvision import datasets, models, transforms

'''
Functions to make images
'''
def make_images(rec,sig,start_ind,end_ind,opath,usetqdm=False):
    matplotlib.use('Agg')
    if usetqdm:
        for i in tqdm(np.arange(start_ind,end_ind)):
            eeg_start = i*2500
            eeg_end = (i+1)*2500        
            if eeg_end < len(sig):
                fname = rec + '_' + str(i) + '.png'
                fpath = os.path.join(opath, fname)
                subeeg = sig[eeg_start:eeg_end]
                fig = plt.figure(figsize=(6.4,4.8))
                plt.plot(np.arange(0,2500),subeeg)
                plt.axis('off')
                fig.savefig(fpath)
                plt.close(fig)
    else:
        for i in np.arange(start_ind,end_ind):
            eeg_start = i*2500
            eeg_end = (i+1)*2500        
            if eeg_end < len(sig):
                fname = rec + '_' + str(i) + '.png'
                fpath = os.path.join(opath, fname)
                subeeg = sig[eeg_start:eeg_end]
                fig = plt.figure(figsize=(6.4,4.8))
                plt.plot(np.arange(0,2500),subeeg)
                plt.axis('off')
                fig.savefig(fpath)
                plt.close(fig)

'''
Predict using torch
'''
# DATASET
class TestDataset(Dataset):
    def __init__(self, datapath):
        self.data_path = datapath
        self.file_names = os.listdir(self.data_path)
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.file_names[index])
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = self.transforms(image)
        img_name = self.file_names[index]
        return image, img_name

# Model
class ResNet_frozen(nn.Module):
    def __init__(self):
        super(ResNet_frozen, self).__init__()
        #res18_modules = list(models.resnet18().children())[:-1]
        self.model = models.resnet18()
        self.model.fc = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,3),
        )
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    # Project path
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Load config
    CONFIG_PATH = os.path.join(dir_path, 'config.yml')
    CONFIG = load_yaml(CONFIG_PATH)

    '''
    1. Make Images
    '''

    # recording path
    ppath = CONFIG['directories']['recordings']
    recordings = os.listdir(ppath)
    recordings = [x for x in recordings if not x.startswith('.')]
    recordings = [x for x in recordings if not x=='images']

    # Make images directory
    img_path = os.path.join(ppath, 'images')
    os.makedirs(img_path, exist_ok=True)
    
    # Make images
    for rec in recordings:
        if not os.path.isfile(os.path.join(ppath, rec, 'remidx_'+rec+'.txt')):
            print('remidx_'+rec+'.txt does not exist')
            if not os.path.isfile(os.path.join(ppath,rec,'sp_'+rec+'.mat')):
                sleepy.calculate_spectrum(ppath, rec, fres=0.5)
            M,S = sleepy.sleep_state(ppath, rec, pwrite=1, pplot=0)
        M,S = sleepy.load_stateidx(ppath, rec)
        # Make images
        eeg1 = np.squeeze(so.loadmat(os.path.join(ppath,rec,'EEG.mat'))['EEG'])
        print('Making images for ' + rec)
        splitlen = int(len(M)/4)
        p1 = Process(target=make_images, args=(rec,eeg1,0,splitlen,img_path))
        p2 = Process(target=make_images, args=(rec,eeg1,splitlen,2*splitlen,img_path))
        p3 = Process(target=make_images, args=(rec,eeg1,2*splitlen,3*splitlen,img_path))
        p4 = Process(target=make_images, args=(rec,eeg1,3*splitlen,len(M),img_path,True))

        p1.start()
        p2.start()
        p3.start()
        p4.start()

        p1.join()
        p2.join()
        p3.join()
        p4.join()

        print('Made images for ' + rec)

        p1.terminate()
        p2.terminate()
        p3.terminate()
        p4.terminate()
    
    '''
    2. Make Predictions
    '''
    # model path
    best_model_path = CONFIG['directories']['best_model']

    # Device
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ResNet_frozen()
    checkpoint = torch.load(best_model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.to(DEVICE)

    # Data loader
    BATCH_SIZE = 8
    NUM_WORKERS = 1
    SHUFFLE = False
    PIN_MEMORY = True
    DROP_LAST = False

    test_dataset = TestDataset(datapath = img_path)
    test_loader = DataLoader(dataset = test_dataset,
                            batch_size = BATCH_SIZE,
                            num_workers = NUM_WORKERS,
                            shuffle = SHUFFLE,
                            pin_memory = PIN_MEMORY,
                            drop_last = DROP_LAST)    
    
    # Make predictions
    print('Making predictions...')
    model.eval()

    y_preds = []
    img_names = []

    for batch_index, (x, img_id) in enumerate(tqdm(test_loader)):
        x = x.to(DEVICE)
        y_logits = model(x).cpu()
        y_pred = torch.argmax(y_logits, dim=1)
        y_pred = y_pred.tolist()
        img_names.extend(img_id)
        y_preds.extend(y_pred)

    print('Writing to remidx...')

    decode_dict = {0:3, 1:1, 2:2}

    indices = [int(x.split('_')[-1].split('.')[0]) for x in img_names]
    recs = [x.split('_')[0]+x.split('_')[1] for x in img_names]
    states = [decode_dict[x] for x in y_preds]

    preddf = pd.DataFrame({'rec':recs, 'idx':indices, 'state':states})

    for rec in recordings:
        subdf = preddf[preddf.rec==rec].reset_index(drop=True)
        subdf = subdf.sort_values(by='idx').reset_index(drop=True)
        M,K = sleepy.load_stateidx(ppath, rec)
        for ind,x in enumerate(subdf.state.values):
            M[ind] = x
        new_remidx(ppath,rec,M,K)

    shutil.rmtree(img_path)
    print('Done!')
