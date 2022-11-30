# !pip install torch
# !pip install mat73
# !pip install tqdm pandas scikit-learn
# !pip install scikit-image
# !pip install wandb -qqq
# !nvidia-smi

import torch
print("Cuda available: ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Cuda version: ", torch.version.cuda)

import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from time import perf_counter
from typing import Callable
import itertools
import mat73
import pandas as pd
import re
import wandb

import sys
import os
import tarfile

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import os.path as op
try:
    from urllib.request import urlretrieve
except ImportError:  # Python 2 compat
    from urllib import urlretrieve  
    
seed = 121
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
print ('Seeds set for training phase')

random_sample = False
print(f'Set random sampling to: {random_sample}')

# class cd:
#     """Context manager for changing the current working directory"""
#     def __init__(self, newPath):
#         self.newPath = os.path.expanduser(newPath)

#     def __enter__(self):
#         self.savedPath = os.getcwd()
#         os.chdir(self.newPath)

#     def __exit__(self, etype, value, traceback):
#         os.chdir(self.savedPath)
        
# cd("./files")

os.chdir("./files/")
file_dir = op.dirname(__file__)
# print(file_dir)
sys.path.append(file_dir)
sys.path.append(os.getcwd())

# print(os.getcwd())


# ./CNPEN/files

from triplet import TripletGenerator, TripletLearner, TripletLoss, TripletLossRaw, RandomTripletGenerator
from builder import create_dataframe, from_tensor_to_numpy, from_numpy_to_tensor, extend_dataframe, build_positive_pairs
from prints import print_img, print_img_from_path, print_img_from_id, print_img_from_classid, print_from_gen, print_from_gen2, print_pair, print_hist_loss, print_hist_dist, print_img_category
from test_train_loops import training, testing, adaptative_train, compute_distances # adaptative_train_lr

print("All librairies correctly imported")

URL = "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
FILENAME = "lfw-deepfunneled.tgz"

if not op.exists(FILENAME):
    print('Downloading %s to %s...' % (URL, FILENAME))
    urlretrieve(URL, FILENAME)

if not op.exists("lfw"):
    print('Extracting image files...')
    tar = tarfile.open("lfw-deepfunneled.tgz")
    tar.extractall("lfw")
    tar.close()
    
PATH = "lfw/lfw-deepfunneled/"

tic = perf_counter()
df_init, all_imgs = create_dataframe()
toc = perf_counter()
print(f"DataFrame creation: {((toc - tic)/60):.1f} min")

tic = perf_counter()
df = extend_dataframe(df_init)
toc = perf_counter()
print(f"DataFrame extention: {((toc - tic)/60):.1f} min")

num_classes = len(df.Classid.unique())

indiv_min = df.Classid.min()
train_valid_percent = 0.75
train_test_percent = 0.8
split_train_valid = int(num_classes * train_valid_percent)
split_train_test = int(num_classes * train_test_percent)
indiv_max = df.Classid.max()

df_train = df[df.Classid<split_train_valid]
df_valid = df[(df.Classid>=split_train_valid)&(df.Classid<split_train_test)]
df_test = df[df.Classid>=split_train_test]

print(f"Train-Valid-Test split: {np.round(100*train_valid_percent)} - {np.round(100*(train_test_percent-train_valid_percent))} - {np.round(100*(1-train_test_percent))} %")

if random_sample == True:
    # -------------
    # For random sampling:
    Xa_train, Xp_train = build_positive_pairs(df, range(indiv_min, split_train_valid))
    Xa_valid, Xp_valid = build_positive_pairs(df, range(split_train_valid, split_train_test))
    Xa_test, Xp_test = build_positive_pairs(df, range(split_train_test, indiv_max-1))
    # Gather the ids of all images that are used for train and test
    all_img_train_idx = list(set(Xa_train) | set(Xp_train))
    all_img_valid_idx = list(set(Xa_train) | set(Xp_train))
    all_img_test_idx = list(set(Xa_test) | set(Xp_test))
    # -------------

value_count = df_train.Classid.value_counts()
value_count = df_valid.Classid.value_counts()
value_count = df_test.Classid.value_counts()

BATCH_SIZE = 128 
BATCH_VALID_SIZE = 8 
BATCH_TEST_SIZE = 32 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Device used: ", device)

model = TripletLearner(base_channels=32, dropout=0)
model.to(device)

print("Model initialized")

lr = 1e-3/2 #1e-3/2
# optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200,300,400,500], gamma=0.5) #milestones=[200,300,400,500] [100,200,300,400] [150,200,250,300,350,400,450]

margin = 0.2
criterion = TripletLoss(margin)
criterion_test = TripletLossRaw(margin)

epochs = 600

print(f"Parameters init: batch_size={BATCH_SIZE}, lr_init={lr}, margin={margin}, nb_epochs={epochs}")

if random_sample == True:
    gen_train = RandomTripletGenerator(all_imgs, Xa_train, Xp_train, BATCH_SIZE, df, all_img_train_idx, device, model, margin, transform = True) #, mining="semi")
    train_loader = DataLoader(gen_train, batch_size=None, shuffle=True)

    gen_valid = RandomTripletGenerator(all_imgs, Xa_valid, Xp_valid, BATCH_VALID_SIZE, df, all_img_valid_idx, device, model, margin)
    valid_loader = DataLoader(gen_valid, batch_size=None, shuffle=True)

    gen_test = RandomTripletGenerator(all_imgs, Xa_test, Xp_test, BATCH_TEST_SIZE, df, all_img_test_idx, device, model, margin)
    test_loader = DataLoader(gen_test, batch_size=None, shuffle=True)

else:
    gen_train = TripletGenerator(df_train, all_imgs, BATCH_SIZE, device, model, margin, transform = True)#, mining='standard')
    train_loader = DataLoader(gen_train, batch_size=None, shuffle=True, num_workers=8)

    gen_valid = TripletGenerator(df_valid, all_imgs, BATCH_VALID_SIZE, device, model, margin)
    valid_loader = DataLoader(gen_valid, batch_size=None, shuffle=True, num_workers=8)

    gen_test = TripletGenerator(df_test, all_imgs, BATCH_TEST_SIZE, device, model, margin)
    test_loader = DataLoader(gen_test, batch_size=None, shuffle=True, num_workers=8)

print("Dataloaders initialized")

wandb.login()

wandb.init(project="triplet_faces",
           name="normalized_600",
           config={"seed" : seed,
                  "batch_size": BATCH_SIZE,
                  "margin": margin,
                  "nb epochs": epochs,
                  "learning_rate" : lr,
                  "scheduler" : [scheduler.milestones,scheduler.gamma],
                  "optimizer" : optimizer,
#                   "criterion" : "euclidean square",
                  "dataset": "LFW",
                  "network_base_channels": model.base_channels,
                  "augment": gen_train.transform,
                   "augmentation": gen_train.apply_augmentation,
                  "dropout": model.dropout,
                  "mining": gen_train.mining})

print("Starting training...")

model = training(model, device, optimizer, scheduler, criterion, epochs, train_loader, valid_loader, save_epoch=False)

torch.save(model.state_dict(), './'+wandb.run.name+'.pth')

print("Model saved")

if wandb.run is not None:
    wandb.finish()
    
torch.cuda.empty_cache()
print("Cleared cache")