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
build_df_fairness, triplet_acc_fairness, bootstrap

seed = 121
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
print ('Seeds set for fairness phase')

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
df_valid = df[(df.Classid>=split_train_valid)&(df.Classid<split_train_test)]
df_test = df[df.Classid>=split_train_test]

BATCH_SIZE = 128 # 128
BATCH_VALID_SIZE = 8 #128 #8
BATCH_TEST_SIZE = 32 #128 #32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
margin = 0.2
criterion = TripletLoss(margin)
criterion_test = TripletLossRaw(margin)

model = TripletLearner(base_channels=32)
model.load_state_dict(torch.load("../models/in_article/base_model.pth",map_location=torch.device('cpu')))
model = model.to(device)
model.eval()

# Build THRESHOLD (for details, see the notebook "determine_threshold")

gen = TripletGenerator(df_valid, all_imgs, BATCH_VALID_SIZE, device, model, margin)
loader = DataLoader(gen, batch_size=None, shuffle=True)

list_loader = []
for _ in range(10):
    list_loader.extend(list(loader))

pos_dist, neg_dist, _ = compute_distances(list_loader, device, model) #loader

y_pos = [1 for _ in range(len(pos_dist))]
y_neg = [0 for _ in range(len(neg_dist))]

y = y_pos + y_neg
X = pos_dist + neg_dist
Xmoins = np.array(X)*(-1)
Xlogistic = np.array(Xmoins).reshape(-1,1)

clf = LogisticRegression(random_state=0).fit(Xlogistic, y)

THRESHOLD = (clf.intercept_/clf.coef_)[0,0]
print("THRESHOLD with logistic regression:", THRESHOLD)

gen = TripletGenerator(df_test, all_imgs, BATCH_TEST_SIZE, device, model, margin, return_id=True)

tic = perf_counter()
df_fairness = build_df_fairness(all_imgs, df_test, gen, 20, device, model, THRESHOLD)
toc = perf_counter()
print(f"DataFrame creation: {((toc - tic)/60):.1f} min")

# GENERAL STATS

print("Same identity - mean distance: ", df_fairness[df_fairness.y_true==1].Distance.mean())
print("Same identity - std distance: ", df_fairness[df_fairness.y_true==1].Distance.std())
print("Same identity - percentiles: ", np.percentile(df_fairness[df_fairness.y_true==1]['Distance'], [5,25,50,75,95]))
print("\n")
print("Different identities - mean distance: ", df_fairness[df_fairness.y_true==0].Distance.mean())
print("Different identities - std distance: ", df_fairness[df_fairness.y_true==0].Distance.std())
print("Different identities - percentiles: ", np.percentile(df_fairness[df_fairness.y_true==0]['Distance'], [5,25,50,75,95]))
print("\n")
print("Mean accuracy: ", df_fairness['correct_predict'].mean())
print("Bootstrapping mean accuracy: ", bootstrap(df_fairness, agg_func=lambda df: df['correct_predict'].mean()))
print("\n")
print("Triplet accuracy: ", triplet_acc_fairness(df_fairness))

pos_dist = df_fairness[df_fairness.y_true==1]['Distance']
neg_dist = df_fairness[df_fairness.y_true==0]['Distance']

X = -np.array(df_fairness.Distance)
y = np.array(df_fairness.y_true)
y_pred = np.array(df_fairness.y_pred)

fpr_dist, tpr_dist, thresholds_dist = roc_curve(y, X)
roc_auc_dist = auc(fpr_dist,tpr_dist)

precision_dist, recall_dist, thresholds_recall_dist = precision_recall_curve(y, X)
auc_s_dist = auc(recall_dist, precision_dist)

tp,fp,fn,tn = confusion_matrix(y, y_pred).ravel()

TPR = tp/(tp+fp)
FPR = fp/(tp+fp)
TNR = tn/(tn+fn)
FNR = fn/(tn+fn)

print("Confusion Matrix Total")
print(confusion_matrix(y, y_pred))

print("\n","Accuracy score:",accuracy_score(y, y_pred))
 
print("\n", "f1 score:", f1_score(y, y_pred), "\n")

print('TPR: ', TPR)
print('FPR: ', FPR)
print('TNR: ', TNR)
print('FNR: ', FNR)

# SUBGROUPS

pos_dist = df_fairness[(df_fairness.y_true==1) & (df_fairness.AB_WhiteMale==1)]['Distance']
neg_dist = df_fairness[(df_fairness.y_true==0) & (df_fairness.AB_WhiteMale==1)]['Distance']
print_hist_dist_zoom(pos_dist, neg_dist, zoom=5.)

pos_dist = df_fairness[(df_fairness.y_true==1) & (df_fairness.AB_NoWhiteMale==1)]['Distance']
neg_dist = df_fairness[(df_fairness.y_true==0) & (df_fairness.AB_NoWhiteMale==1)]['Distance']
print_hist_dist_zoom(pos_dist, neg_dist, zoom=5.)

pos_dist = df_fairness[(df_fairness.y_true==1) & (df_fairness.AB_WhiteMale==0)]['Distance']
neg_dist = df_fairness[(df_fairness.y_true==0) & (df_fairness.AB_WhiteMale==0)]['Distance']
print_hist_dist_zoom(pos_dist, neg_dist, zoom=5.)