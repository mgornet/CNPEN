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
    
from sklearn.metrics import confusion_matrix, auc, roc_curve, \
precision_recall_curve, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from scipy import optimize

import wandb

os.chdir("./files/")
file_dir = op.dirname(__file__)

sys.path.append(file_dir)
sys.path.append(os.getcwd())

from triplet import TripletGenerator, TripletLearner, TripletLoss, TripletLossRaw, \
distance, distance_vectors
from builder import create_dataframe, from_tensor_to_numpy, from_numpy_to_tensor, extend_dataframe
from prints import print_img, print_img_from_path, print_img_from_id, \
print_img_from_classid, print_from_gen, print_from_gen2, print_pair, print_hist_loss, \
print_hist_dist, print_hist_dist_zoom, print_img_category, \
print_roc, print_logistic_regression, print_prec_recall
from test_train_loops import training, testing, adaptative_train, compute_distances
from classification import authentification_img, predict, triplet_acc,\
build_df_fairness, triplet_acc_fairness, bootstrap, bootstrap_by_pairs, build_threshold

# import warnings
# warnings.filterwarnings("ignore")

seed = 121
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
print ('Seeds set for testing phase')

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
print("Number of individuals: ", num_classes)

indiv_min = df.Classid.min()
split_train_valid = int(num_classes * 0.75)
split_train_test = int(num_classes * 0.8)
indiv_max = df.Classid.max()

df_train = df[df.Classid<split_train_valid]
df_valid = df[(df.Classid>=split_train_valid)]#&(df.Classid<split_train_test)]
# df_test = df[df.Classid>=split_train_test]
df_test = df_valid

BATCH_SIZE = 128 # 128
BATCH_VALID_SIZE = 128 #128 #8
BATCH_TEST_SIZE = 128 #128 #32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
margin = 0.2
# criterion = TripletLoss(margin)
# criterion_test = TripletLossRaw(margin)

# Load models

model_base = TripletLearner(base_channels=32)
model_base.load_state_dict(torch.load("../models/in_article/base_600.pth",map_location=torch.device('cpu'))) #../models/in_article/base_model
model_base = model_base.to(device)
model_base.eval()

# Create generator and df_fairness
    
gen_base = TripletGenerator(df_test, all_imgs, BATCH_TEST_SIZE, device, model_base, margin, return_id=True)
threshold_base  = build_threshold(df_valid, all_imgs, device, model_base, margin, True)
print(f"Threshold: {np.round(threshold_base)}")
tic = perf_counter()
df_fairness = build_df_fairness(all_imgs, df_test, gen_base, 20, device, model_base, threshold_base) #20
toc = perf_counter()
print(f"DataFrame creation: {((toc - tic)/60):.1f} min")

# RESULTS
print("\n")
b,a,c = np.round_(bootstrap(df_fairness, agg_func=lambda df: df['correct_predict'].mean(), num_bootstraps=1000, percentiles=[5,50,95]), 3)
print(f"Accuracy: {a} ({b}-{c})")
b,a,c = np.round_(bootstrap_by_pairs(df_fairness, agg_func=lambda df: triplet_acc_fairness(df), num_bootstraps=1000, percentiles=[5,50,95]), 3)
print(f"Triplet accuracy: {a} ({b}-{c})")
b,a,c = np.round_(bootstrap(df_fairness, agg_func=lambda df: df[(df['y_pred']==1)&(df['y_true']==0)]['id_A'].count()/df[df['y_pred']==1]['id_A'].count(), num_bootstraps=1000, percentiles=[5,50,95]),3)
print(f"FPR: {a} ({b}-{c})")
b,a,c = np.round_(bootstrap(df_fairness, agg_func=lambda df: df[(df['y_pred']==0)&(df['y_true']==1)]['id_A'].count()/df[df['y_pred']==0]['id_A'].count(), num_bootstraps=1000, percentiles=[5,50,95]),3)
print(f"FNR: {a} ({b}-{c})")
print("\n")


torch.cuda.empty_cache()
print("Cleared cache")