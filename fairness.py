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
build_df_fairness, triplet_acc_fairness, bootstrap, bootstrap_by_pairs

# import warnings
# warnings.filterwarnings("ignore")

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

# Build THRESHOLD (for details, see the notebook "determine_threshold")
def build_threshold(df_valid, all_imgs, device, model, margin, verbose=False):
    
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

    threshold = (clf.intercept_/clf.coef_)[0,0]
    
    if verbose:
        print("Threshold with logistic regression:", threshold)

    return threshold

# Load models

model_base = TripletLearner(base_channels=32)
model_base.load_state_dict(torch.load("../models/in_article/base_121_600.pth",map_location=torch.device('cpu'))) #../models/in_article/base_model
model_base = model_base.to(device)
model_base.eval()

model_margin1 = TripletLearner(base_channels=32)
model_margin1.load_state_dict(torch.load("../models/in_article/margin1_1000epochs.pth",map_location=torch.device('cpu')))
model_margin1 = model_margin1.to(device)
model_margin1.eval()

model_margin05 = TripletLearner(base_channels=32)
model_margin05.load_state_dict(torch.load("../models/in_article/margin05.pth",map_location=torch.device('cpu')))
model_margin05 = model_margin05.to(device)
model_margin05.eval()

model_margin01 = TripletLearner(base_channels=32)
model_margin01.load_state_dict(torch.load("../models/in_article/margin01.pth",map_location=torch.device('cpu')))
model_margin01 = model_margin01.to(device)
model_margin01.eval()

model_jitterbrightness = TripletLearner(base_channels=32)
model_jitterbrightness.load_state_dict(torch.load("../models/in_article/high_brightness_600.pth",map_location=torch.device('cpu'))) #../models/in_article/base_model
model_jitterbrightness = model_base.to(device)
model_jitterbrightness.eval()

# Create generator and df_fairness
    
gen_base = TripletGenerator(df_test, all_imgs, BATCH_TEST_SIZE, device, model_base, margin, return_id=True)
threshold_base  = build_threshold(df_valid, all_imgs, device, model_base, margin, True)
tic = perf_counter()
df_fairness_base = build_df_fairness(all_imgs, df_test, gen_base, 20, device, model_base, threshold_base) #20
toc = perf_counter()
print(f"DataFrame creation: {((toc - tic)/60):.1f} min")

gen_margin1 = TripletGenerator(df_test, all_imgs, BATCH_TEST_SIZE, device, model_margin1, 1., return_id=True)
threshold_margin1  = build_threshold(df_valid, all_imgs, device, model_margin1, 1., True)
tic = perf_counter()
df_fairness_margin1 = build_df_fairness(all_imgs, df_test, gen_margin1, 20, device, model_margin1, threshold_margin1)
toc = perf_counter()
print(f"DataFrame creation: {((toc - tic)/60):.1f} min")

gen_margin05 = TripletGenerator(df_test, all_imgs, BATCH_TEST_SIZE, device, model_margin05, 0.5, return_id=True)
threshold_margin05  = build_threshold(df_valid, all_imgs, device, model_margin05, 0.5, True)
tic = perf_counter()
df_fairness_margin05 = build_df_fairness(all_imgs, df_test, gen_margin05, 20, device, model_margin05, threshold_margin05)
toc = perf_counter()
print(f"DataFrame creation: {((toc - tic)/60):.1f} min")

gen_margin01 = TripletGenerator(df_test, all_imgs, BATCH_TEST_SIZE, device, model_margin01, 0.1, return_id=True)
threshold_margin01  = build_threshold(df_valid, all_imgs, device, model_margin01, 0.1, True)
tic = perf_counter()
df_fairness_margin01 = build_df_fairness(all_imgs, df_test, gen_margin01, 20, device, model_margin01, threshold_margin01)
toc = perf_counter()
print(f"DataFrame creation: {((toc - tic)/60):.1f} min")

gen_jitterbrightness = TripletGenerator(df_test, all_imgs, BATCH_TEST_SIZE, device, model_jitterbrightness, margin, return_id=True)
threshold_jitterbrightness  = build_threshold(df_valid, all_imgs, device, model_jitterbrightness, margin, True)
tic = perf_counter()
df_fairness_jitterbrightness = build_df_fairness(all_imgs, df_test, gen_jitterbrightness, 20, device, model_jitterbrightness, threshold_jitterbrightness) #20
toc = perf_counter()
print(f"DataFrame creation: {((toc - tic)/60):.1f} min")



# RESULTS

def results(df_fairness, analysis):
    
    # Male vs Non Male
    if analysis == "gender":

        b,a,c = np.round_(bootstrap(df_fairness[(df_fairness['A_Male']==1)&(df_fairness['B_Male']==1)], \
            agg_func=lambda df: df['correct_predict'].mean(), num_bootstraps=1000, percentiles=[5,50,95]), 3)
        print(f"Male - accuracy: {a} ({b}-{c})")
        b,a,c = np.round_(bootstrap_by_pairs(df_fairness[(df_fairness['A_Male']==1)&(df_fairness['B_Male']==1)], \
            agg_func=lambda df: triplet_acc_fairness(df), num_bootstraps=1000, percentiles=[5,50,95]), 3)
        print(f"Male - triplet accuracy: {a} ({b}-{c})")
        b,a,c = np.round_(bootstrap(df_fairness[(df_fairness['A_Male']==1)&(df_fairness['B_Male']==1)], \
            agg_func=lambda df: df[(df['y_pred']==1)&(df['y_true']==0)]['id_A'].count()/df[df['y_pred']==1]['id_A'].count(), \
            num_bootstraps=1000, percentiles=[5,50,95]),3)
        print(f"Male - FPR: {a} ({b}-{c})")
        b,a,c = np.round_(bootstrap(df_fairness[(df_fairness['A_Male']==1)&(df_fairness['B_Male']==1)], \
            agg_func=lambda df: df[(df['y_pred']==0)&(df['y_true']==1)]['id_A'].count()/df[df['y_pred']==0]['id_A'].count(), \
            num_bootstraps=1000, percentiles=[5,50,95]),3)
        print(f"Male - FNR: {a} ({b}-{c})")
        print("\n")
        b,a,c = np.round_(bootstrap(df_fairness[(df_fairness['A_Male']==0)&(df_fairness['B_Male']==0)], \
            agg_func=lambda df: df['correct_predict'].mean(), num_bootstraps=1000, percentiles=[5,50,95]),3)
        print(f"Non Male - accuracy: {a} ({b}-{c})")
        b,a,c = np.round_(bootstrap_by_pairs(df_fairness[(df_fairness['A_Male']==0)&(df_fairness['B_Male']==0)], \
            agg_func=lambda df: triplet_acc_fairness(df), num_bootstraps=1000, percentiles=[5,50,95]), 3)
        print(f"Non Male - triplet accuracy: {a} ({b}-{c})")
        b,a,c = np.round_(bootstrap(df_fairness[(df_fairness['A_Male']==0)&(df_fairness['B_Male']==0)], \
            agg_func=lambda df: df[(df['y_pred']==1)&(df['y_true']==0)]['id_A'].count()/df[df['y_pred']==1]['id_A'].count(), \
            num_bootstraps=1000, percentiles=[5,50,95]),3)
        print(f"Non Male - FPR: {a} ({b}-{c})")
        b,a,c = np.round_(bootstrap(df_fairness[(df_fairness['A_Male']==0)&(df_fairness['B_Male']==0)], \
            agg_func=lambda df: df[(df['y_pred']==0)&(df['y_true']==1)]['id_A'].count()/df[df['y_pred']==0]['id_A'].count(), \
            num_bootstraps=1000, percentiles=[5,50,95]),3)
        print(f"Non Male - FNR: {a} ({b}-{c})")
        print("\n")
        
    elif analysis == "color":
        
        b,a,c = np.round_(bootstrap(df_fairness[(df_fairness['A_White']==1)&(df_fairness['B_White']==1)], \
            agg_func=lambda df: df['correct_predict'].mean(), num_bootstraps=1000, percentiles=[5,50,95]), 3)
        print(f"White - accuracy: {a} ({b}-{c})")
        b,a,c = np.round_(bootstrap_by_pairs(df_fairness[(df_fairness['A_White']==1)&(df_fairness['B_White']==1)], \
            agg_func=lambda df: triplet_acc_fairness(df), num_bootstraps=1000, percentiles=[5,50,95]), 3)
        print(f"White - triplet accuracy: {a} ({b}-{c})")
        b,a,c = np.round_(bootstrap(df_fairness[(df_fairness['A_White']==1)&(df_fairness['B_White']==1)], \
            agg_func=lambda df: df[(df['y_pred']==1)&(df['y_true']==0)]['id_A'].count()/df[df['y_pred']==1]['id_A'].count(), \
            num_bootstraps=1000, percentiles=[5,50,95]),3)
        print(f"White - FPR: {a} ({b}-{c})")
        b,a,c = np.round_(bootstrap(df_fairness[(df_fairness['A_White']==1)&(df_fairness['B_White']==1)], \
            agg_func=lambda df: df[(df['y_pred']==0)&(df['y_true']==1)]['id_A'].count()/df[df['y_pred']==0]['id_A'].count(), \
            num_bootstraps=1000, percentiles=[5,50,95]),3)
        print(f"White - FNR: {a} ({b}-{c})")
        print("\n")
        b,a,c = np.round_(bootstrap(df_fairness[(df_fairness['A_White']==0)&(df_fairness['B_White']==0)], \
            agg_func=lambda df: df['correct_predict'].mean(), num_bootstraps=1000, percentiles=[5,50,95]), 3)
        print(f"Non-White - accuracy: {a} ({b}-{c})")
        b,a,c = np.round_(bootstrap_by_pairs(df_fairness[(df_fairness['A_White']==0)&(df_fairness['B_White']==0)], \
            agg_func=lambda df: triplet_acc_fairness(df), num_bootstraps=1000, percentiles=[5,50,95]), 3)
        print(f"Non-White - triplet accuracy: {a} ({b}-{c})")
        b,a,c = np.round_(bootstrap(df_fairness[(df_fairness['A_White']==0)&(df_fairness['B_White']==0)], \
            agg_func=lambda df: df[(df['y_pred']==1)&(df['y_true']==0)]['id_A'].count()/df[df['y_pred']==1]['id_A'].count(), \
            num_bootstraps=1000, percentiles=[5,50,95]),3)
        print(f"Non-White - FPR: {a} ({b}-{c})")
        b,a,c = np.round_(bootstrap(df_fairness[(df_fairness['A_White']==0)&(df_fairness['B_White']==0)], \
            agg_func=lambda df: df[(df['y_pred']==0)&(df['y_true']==1)]['id_A'].count()/df[df['y_pred']==0]['id_A'].count(), \
            num_bootstraps=1000, percentiles=[5,50,95]),3)
        print(f"Non-White - FNR: {a} ({b}-{c})")
        print("\n")
        b,a,c = np.round_(bootstrap(df_fairness[(df_fairness['A_Black']==1)&(df_fairness['B_Black']==1)], \
            agg_func=lambda df: df['correct_predict'].mean(), num_bootstraps=1000, percentiles=[5,50,95]),3)
        print(f"Black - accuracy: {a} ({b}-{c})")
        b,a,c = np.round_(bootstrap_by_pairs(df_fairness[(df_fairness['A_Black']==1)&(df_fairness['B_Black']==1)], \
            agg_func=lambda df: triplet_acc_fairness(df), num_bootstraps=1000, percentiles=[5,50,95]), 3)
        print(f"Black - triplet accuracy: {a} ({b}-{c})")
        b,a,c = np.round_(bootstrap(df_fairness[(df_fairness['A_Black']==1)&(df_fairness['B_Black']==1)], \
            agg_func=lambda df: df[(df['y_pred']==1)&(df['y_true']==0)]['id_A'].count()/df[df['y_pred']==1]['id_A'].count(), \
            num_bootstraps=1000, percentiles=[5,50,95]),3)
        print(f"Black - FPR: {a} ({b}-{c})")
        b,a,c = np.round_(bootstrap(df_fairness[(df_fairness['A_Black']==1)&(df_fairness['B_Black']==1)], \
            agg_func=lambda df: df[(df['y_pred']==0)&(df['y_true']==1)]['id_A'].count()/df[df['y_pred']==0]['id_A'].count(), \
            num_bootstraps=1000, percentiles=[5,50,95]),3)
        print(f"Black - FNR: {a} ({b}-{c})")
        # print("\n")
        # b,a,c = np.round_(bootstrap(df_fairness[(df_fairness['A_Asian']==1)&(df_fairness['B_Asian']==1)], \
        #     agg_func=lambda df: df['correct_predict'].mean(), num_bootstraps=10000, percentiles=[5,50,95]), 3)
        # print(f"Asian - accuracy: {a} ({b}-{c})")
        # b,a,c = np.round_(bootstrap_by_pairs(df_fairness[(df_fairness['A_Asian']==1)&(df_fairness['B_Asian']==1)], \
        #     agg_func=lambda df: triplet_acc_fairness(df), num_bootstraps=100, percentiles=[5,50,95]), 3)
        # print(f"Asian - triplet accuracy: {a} ({b}-{c})")
        # b,a,c = np.round_(bootstrap(df_fairness[(df_fairness['A_Asian']==1)&(df_fairness['B_Asian']==1)], \
        #     agg_func=lambda df: df[(df['y_pred']==1)&(df['y_true']==0)]['id_A'].count()/df[df['y_pred']==1]['id_A'].count(), \
        #     num_bootstraps=10000, percentiles=[5,50,95]),3)
        # print(f"Asian - FPR: {a} ({b}-{c})")
        # b,a,c = np.round_(bootstrap(df_fairness[(df_fairness['A_Asian']==1)&(df_fairness['B_Asian']==1)], \
        #     agg_func=lambda df: df[(df['y_pred']==0)&(df['y_true']==1)]['id_A'].count()/df[df['y_pred']==0]['id_A'].count(), \
        #     num_bootstraps=10000, percentiles=[5,50,95]),3)
        # print(f"Asian - FNR: {a} ({b}-{c})")
        # print("\n")
        # b,a,c = np.round_(bootstrap(df_fairness[(df_fairness['A_Indian']==1)&(df_fairness['B_Indian']==1)], \
        #     agg_func=lambda df: df['correct_predict'].mean(), num_bootstraps=10000, percentiles=[5,50,95]),3)
        # print(f"Indian - accuracy: {a} ({b}-{c})")
        # b,a,c = np.round_(bootstrap_by_pairs(df_fairness[(df_fairness['A_Indian']==1)&(df_fairness['B_Indian']==1)], \
        #     agg_func=lambda df: triplet_acc_fairness(df), num_bootstraps=100, percentiles=[5,50,95]), 3)
        # print(f"Indian - triplet accuracy: {a} ({b}-{c})")
        # b,a,c = np.round_(bootstrap(df_fairness[(df_fairness['A_Indian']==1)&(df_fairness['B_Indian']==1)], \
        #     agg_func=lambda df: df[(df['y_pred']==1)&(df['y_true']==0)]['id_A'].count()/df[df['y_pred']==1]['id_A'].count(), \
        #     num_bootstraps=10000, percentiles=[5,50,95]),3)
        # print(f"Indian - FPR: {a} ({b}-{c})")
        # b,a,c = np.round_(bootstrap(df_fairness[(df_fairness['A_Indian']==1)&(df_fairness['B_Indian']==1)], \
        #     agg_func=lambda df: df[(df['y_pred']==0)&(df['y_true']==1)]['id_A'].count()/df[df['y_pred']==0]['id_A'].count(), \
        #     num_bootstraps=10000, percentiles=[5,50,95]),3)
        # print(f"Indian - FNR: {a} ({b}-{c})")
        print("\n")
        
    elif analysis == "intersectional":
    
        b,a,c = np.round_(bootstrap(df_fairness[df_fairness['AB_WhiteMale']==1], \
        agg_func=lambda df: df['correct_predict'].mean(), num_bootstraps=1000, percentiles=[5,50,95]), 3)
        print(f"White Male - accuracy: {a} (90% CI: {b}-{c})")
        b,a,c = np.round_(bootstrap_by_pairs(df_fairness[df_fairness['AB_WhiteMale']==1], \
            agg_func=lambda df: triplet_acc_fairness(df), num_bootstraps=1000, percentiles=[5,50,95]), 3)
        print(f"White Male - triplet accuracy: {a} (90% CI: {b}-{c})")
        b,a,c = np.round_(bootstrap(df_fairness[df_fairness['AB_WhiteMale']==1], \
            agg_func=lambda df: df[(df['y_pred']==1)&(df['y_true']==0)]['id_A'].count()/df[df['y_pred']==1]['id_A'].count(), \
            num_bootstraps=1000, percentiles=[5,50,95]),3)
        print(f"White Male - FPR: {a} (90% CI: {b}-{c})")
        b,a,c = np.round_(bootstrap(df_fairness[df_fairness['AB_WhiteMale']==1], \
            agg_func=lambda df: df[(df['y_pred']==0)&(df['y_true']==1)]['id_A'].count()/df[df['y_pred']==0]['id_A'].count(), \
            num_bootstraps=1000, percentiles=[5,50,95]),3)
        print(f"White Male - FNR: {a} (90% CI: {b}-{c})")
        print("\n")
        b,a,c = np.round_(bootstrap(df_fairness[df_fairness['AB_NoWhiteMale']==1], \
            agg_func=lambda df: df['correct_predict'].mean(), num_bootstraps=1000, percentiles=[5,50,95]),3)
        print(f"Non-White Female - accuracy: {a} (90% CI: {b}-{c})")
        b,a,c = np.round_(bootstrap_by_pairs(df_fairness[df_fairness['AB_NoWhiteMale']==1], \
            agg_func=lambda df: triplet_acc_fairness(df), num_bootstraps=1000, percentiles=[5,50,95]), 3)
        print(f"Non-White Female - triplet accuracy: {a} (90% CI: {b}-{c})")
        b,a,c = np.round_(bootstrap(df_fairness[df_fairness['AB_NoWhiteMale']==1], \
            agg_func=lambda df: df[(df['y_pred']==1)&(df['y_true']==0)]['id_A'].count()/df[df['y_pred']==1]['id_A'].count(), \
            num_bootstraps=1000, percentiles=[5,50,95]),3)
        print(f"Non-White Female - FPR: {a} (90% CI: {b}-{c})")
        b,a,c = np.round_(bootstrap(df_fairness[df_fairness['AB_NoWhiteMale']==1], \
            agg_func=lambda df: df[(df['y_pred']==0)&(df['y_true']==1)]['id_A'].count()/df[df['y_pred']==0]['id_A'].count(), \
            num_bootstraps=1000, percentiles=[5,50,95]),3)
        print(f"Non-White Female - FNR: {a} (90% CI: {b}-{c})")
        print("\n")
        b,a,c = np.round_(bootstrap(df_fairness[(df_fairness['A_Black']==1)&(df_fairness['B_Black']==1)&(df_fairness['A_Male']==0)&(df_fairness['B_Male']==0)], \
            agg_func=lambda df: df['correct_predict'].mean(), num_bootstraps=1000, percentiles=[5,50,95]),3)
        print(f"Black Female - accuracy: {a} (90% CI: {b}-{c})")
        b,a,c = np.round_(bootstrap_by_pairs(df_fairness[(df_fairness['A_Black']==1)&(df_fairness['B_Black']==1)&(df_fairness['A_Male']==0)&(df_fairness['B_Male']==0)], \
            agg_func=lambda df: triplet_acc_fairness(df), num_bootstraps=1000, percentiles=[5,50,95]), 3)
        print(f"Black Female - triplet accuracy: {a} (90% CI: {b}-{c})")
        b,a,c = np.round_(bootstrap(df_fairness[(df_fairness['A_Black']==1)&(df_fairness['B_Black']==1)&(df_fairness['A_Male']==0)&(df_fairness['B_Male']==0)], \
            agg_func=lambda df: df[(df['y_pred']==1)&(df['y_true']==0)]['id_A'].count()/df[df['y_pred']==1]['id_A'].count(), \
            num_bootstraps=1000, percentiles=[5,50,95]),3)
        print(f"Black Female - FPR: {a} (90% CI: {b}-{c})")
        b,a,c = np.round_(bootstrap(df_fairness[(df_fairness['A_Black']==1)&(df_fairness['B_Black']==1)&(df_fairness['A_Male']==0)&(df_fairness['B_Male']==0)], \
            agg_func=lambda df: df[(df['y_pred']==0)&(df['y_true']==1)]['id_A'].count()/df[df['y_pred']==0]['id_A'].count(), \
            num_bootstraps=1000, percentiles=[5,50,95]),3)
        print(f"Black Female - FNR: {a} (90% CI: {b}-{c})")
        print("\n")

        
    else:
        print("Please select a correct analysis")

        
print("\n") 


print("------------------------------------------------")
print("GENDER ANALYSIS")
print("------------------------------------------------")
print("\n")


# print("Margin 1")
# results(df_fairness_margin1, "gender")

# print("Margin 0.5")
# results(df_fairness_margin05, "gender")

print("Margin 0.2")
results(df_fairness_base, "gender")

# print("Margin 0.1")
# results(df_fairness_margin01, "gender")

print("High brightness")
results(df_fairness_jitterbrightness, "gender")
                                                
print("------------------------------------------------")
print("SKIN COLOR ANALYSIS")
print("------------------------------------------------")
print("\n")                                               

# print("Margin 1")
# results(df_fairness_margin1, "color")

# print("Margin 0.5")
# results(df_fairness_margin05, "color")

print("Margin 0.2")
results(df_fairness_base, "color")

# print("Margin 0.1")
# results(df_fairness_margin01, "color")

print("High brightness")
results(df_fairness_jitterbrightness, "color")

print("------------------------------------------------")
print("INTERSECTIONAL ANALYSIS")
print("------------------------------------------------")
print("\n")
                                                
# print("Margin 1")
# results(df_fairness_margin1, "intersectional")

# print("Margin 0.5")
# results(df_fairness_margin05, "intersectional")

print("Margin 0.2")
results(df_fairness_base, "intersectional")

# print("Margin 0.1")
# results(df_fairness_margin01, "intersectional")

print("High brightness")
results(df_fairness_jitterbrightness, "intersectional")


torch.cuda.empty_cache()
print("Cleared cache")