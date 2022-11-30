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

print("All librairies correctly imported")

###################################
# CHANGE HERE
###################################
study = 'augmentation'
if study not in ['sampling','augmentation', 'normalization', 'depth', 'margin', 'lr', 'threshold']:
    print("The study name is incorrect, please select between: sampling, normalization, augmentation, depth, margin, lr, threshold")
    
# If using normalization, please do not forget to change the open_one_image_tensor function in the builder file to normalize the test set

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
BATCH_VALID_SIZE = 59 #128 #8
BATCH_TEST_SIZE = 59 #128 #32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
margin = 0.2
# criterion = TripletLoss(margin)
# criterion_test = TripletLossRaw(margin)

# Load models, create generator and df_fairness

model_base = TripletLearner(base_channels=32)
model_base.load_state_dict(torch.load("../models/in_article/base_600.pth",map_location=torch.device('cpu'))) #../models/in_article/base_model
model_base = model_base.to(device)
model_base.eval()

gen_base = TripletGenerator(df_test, all_imgs, BATCH_TEST_SIZE, device, model_base, margin, return_id=True)
threshold_base  = build_threshold(df_valid, all_imgs, device, model_base, margin, BATCH_VALID_SIZE, True)
tic = perf_counter()
df_fairness_base = build_df_fairness(all_imgs, df_test, gen_base, 20, device, model_base, threshold_base) #20
toc = perf_counter()
print(f"DataFrame creation for base model: {((toc - tic)/60):.1f} min")

# df_1 = df_fairness_base[(df_fairness_base['A_Black']==1)&(df_fairness_base['B_Black']==1)&(df_fairness_base['A_Male']==0)&(df_fairness_base['B_Male']==0)]
# print(df_1[df_1['y_pred']==0]['id_A'].count())
# print(df_1[df_1['y_pred']==1]['id_A'].count())
# print(df_1[(df_1['y_pred']==0)&(df_1['y_true']==1)]['id_A'].count())

if study == 'sampling':
    
    model_randomsample = TripletLearner(base_channels=32)
    model_randomsample.load_state_dict(torch.load("../models/in_article/random_sample_600.pth",map_location=torch.device('cpu')))
    model_randomsample = model_randomsample.to(device)
    model_randomsample.eval()
    
    gen_randomsample = TripletGenerator(df_test, all_imgs, BATCH_TEST_SIZE, device, model_randomsample, 0.2, return_id=True)
    threshold_randomsample  = build_threshold(df_valid, all_imgs, device, model_randomsample, 0.2, BATCH_VALID_SIZE, True)
    tic = perf_counter()
    df_fairness_randomsample = build_df_fairness(all_imgs, df_test, gen_randomsample, 20, device, model_randomsample, threshold_randomsample)
    toc = perf_counter()
    print(f"DataFrame creation for model random sampling: {((toc - tic)/60):.1f} min")
    
elif study == 'normalization':
    
    model_normalized = TripletLearner(base_channels=32)
    model_normalized.load_state_dict(torch.load("../models/in_article/normalized_600.pth",map_location=torch.device('cpu')))
    model_normalized = model_normalized.to(device)
    model_normalized.eval()
    
    gen_normalized = TripletGenerator(df_test, all_imgs, BATCH_TEST_SIZE, device, model_normalized, 0.2, return_id=True)
    threshold_normalized  = build_threshold(df_valid, all_imgs, device, model_normalized, 0.2, BATCH_VALID_SIZE, True)
    tic = perf_counter()
    df_fairness_normalized = build_df_fairness(all_imgs, df_test, gen_normalized, 20, device, model_normalized, threshold_normalized)
    toc = perf_counter()
    print(f"DataFrame creation for model random sampling: {((toc - tic)/60):.1f} min")
    
elif study == 'augmentation':
    model_no_augment = TripletLearner(base_channels=32)
    model_no_augment.load_state_dict(torch.load("../models/in_article/no_augment_600.pth",map_location=torch.device('cpu')))
    model_no_augment = model_no_augment.to(device)
    model_no_augment.eval()
    
    gen_no_augment = TripletGenerator(df_test, all_imgs, BATCH_TEST_SIZE, device, model_no_augment, 0.2, return_id=True)
    threshold_no_augment  = build_threshold(df_valid, all_imgs, device, model_no_augment, 0.2, BATCH_VALID_SIZE, True)
    tic = perf_counter()
    df_fairness_no_augment = build_df_fairness(all_imgs, df_test, gen_no_augment, 20, device, model_no_augment, threshold_no_augment)
    toc = perf_counter()
    print(f"DataFrame creation for model no augmentation: {((toc - tic)/60):.1f} min")
    
    model_high_zoom = TripletLearner(base_channels=32)
    model_high_zoom.load_state_dict(torch.load("../models/in_article/high_zoom_600.pth",map_location=torch.device('cpu')))
    model_high_zoom = model_high_zoom.to(device)
    model_high_zoom.eval()
    
    gen_high_zoom = TripletGenerator(df_test, all_imgs, BATCH_TEST_SIZE, device, model_high_zoom, 0.2, return_id=True)
    threshold_high_zoom  = build_threshold(df_valid, all_imgs, device, model_high_zoom, 0.2, BATCH_VALID_SIZE, True)
    tic = perf_counter()
    df_fairness_high_zoom = build_df_fairness(all_imgs, df_test, gen_high_zoom, 20, device, model_high_zoom, threshold_high_zoom)
    toc = perf_counter()
    print(f"DataFrame creation for model high zoom: {((toc - tic)/60):.1f} min")
    
    model_deformation_double = TripletLearner(base_channels=32)
    model_deformation_double.load_state_dict(torch.load("../models/in_article/deformation_double_600.pth",map_location=torch.device('cpu')))
    model_deformation_double = model_deformation_double.to(device)
    model_deformation_double.eval()
    
    gen_deformation_double = TripletGenerator(df_test, all_imgs, BATCH_TEST_SIZE, device, model_deformation_double, 0.2, return_id=True)
    threshold_deformation_double  = build_threshold(df_valid, all_imgs, device, model_deformation_double, 0.2, BATCH_VALID_SIZE, True)
    tic = perf_counter()
    df_fairness_deformation_double = build_df_fairness(all_imgs, df_test, gen_deformation_double, 20, device, model_deformation_double, threshold_deformation_double)
    toc = perf_counter()
    print(f"DataFrame creation for model high deformation: {((toc - tic)/60):.1f} min")
    
    model_jitter_double = TripletLearner(base_channels=32)
    model_jitter_double.load_state_dict(torch.load("../models/in_article/jitter_double_600.pth",map_location=torch.device('cpu')))
    model_jitter_double = model_jitter_double.to(device)
    model_jitter_double.eval()
    
    gen_jitter_double = TripletGenerator(df_test, all_imgs, BATCH_TEST_SIZE, device, model_jitter_double, 0.2, return_id=True)
    threshold_jitter_double  = build_threshold(df_valid, all_imgs, device, model_jitter_double, 0.2, BATCH_VALID_SIZE, True)
    tic = perf_counter()
    df_fairness_jitter_double = build_df_fairness(all_imgs, df_test, gen_jitter_double, 20, device, model_jitter_double, threshold_jitter_double)
    toc = perf_counter()
    print(f"DataFrame creation for model high color jitter: {((toc - tic)/60):.1f} min")
    
    model_high_rotation = TripletLearner(base_channels=32)
    model_high_rotation.load_state_dict(torch.load("../models/in_article/high_rotation_600.pth",map_location=torch.device('cpu')))
    model_high_rotation = model_high_rotation.to(device)
    model_high_rotation.eval()
    
    gen_high_rotation = TripletGenerator(df_test, all_imgs, BATCH_TEST_SIZE, device, model_high_rotation, 0.2, return_id=True)
    threshold_high_rotation  = build_threshold(df_valid, all_imgs, device, model_high_rotation, 0.2, BATCH_VALID_SIZE, True)
    tic = perf_counter()
    df_fairness_high_rotation = build_df_fairness(all_imgs, df_test, gen_high_rotation, 20, device, model_high_rotation, threshold_high_rotation)
    toc = perf_counter()
    print(f"DataFrame creation for model high rotation: {((toc - tic)/60):.1f} min")
    
    
elif study == 'depth':
    model_depth16 = TripletLearner(base_channels=16)
    model_depth16.load_state_dict(torch.load("../models/in_article/depth16_600.pth",map_location=torch.device('cpu')))
    model_depth16 = model_depth16.to(device)
    model_depth16.eval()
    
    gen_depth16 = TripletGenerator(df_test, all_imgs, BATCH_TEST_SIZE, device, model_depth16, 0.2, return_id=True)
    threshold_depth16  = build_threshold(df_valid, all_imgs, device, model_depth16, 0.2, BATCH_VALID_SIZE, True)
    tic = perf_counter()
    df_fairness_depth16 = build_df_fairness(all_imgs, df_test, gen_depth16, 20, device, model_depth16, threshold_depth16)
    toc = perf_counter()
    print(f"DataFrame creation for model depth 16: {((toc - tic)/60):.1f} min")
    
    model_depth64 = TripletLearner(base_channels=64)
    model_depth64.load_state_dict(torch.load("../models/in_article/depth64_600.pth",map_location=torch.device('cpu')))
    model_depth64 = model_depth64.to(device)
    model_depth64.eval()
    
    gen_depth64 = TripletGenerator(df_test, all_imgs, BATCH_TEST_SIZE, device, model_depth64, 0.2, return_id=True)
    threshold_depth64  = build_threshold(df_valid, all_imgs, device, model_depth64, 0.2, BATCH_VALID_SIZE, True)
    tic = perf_counter()
    df_fairness_depth64 = build_df_fairness(all_imgs, df_test, gen_depth64, 20, device, model_depth64, threshold_depth64)
    toc = perf_counter()
    print(f"DataFrame creation for model depth 64: {((toc - tic)/60):.1f} min")

elif study == 'margin':
    
    model_margin01 = TripletLearner(base_channels=32)
    model_margin01.load_state_dict(torch.load("../models/in_article/margin01_600.pth",map_location=torch.device('cpu')))
    model_margin01 = model_margin01.to(device)
    model_margin01.eval()
    
    gen_margin01 = TripletGenerator(df_test, all_imgs, BATCH_TEST_SIZE, device, model_margin01, 0.1, return_id=True)
    threshold_margin01  = build_threshold(df_valid, all_imgs, device, model_margin01, 0.1, BATCH_VALID_SIZE, True)
    tic = perf_counter()
    df_fairness_margin01 = build_df_fairness(all_imgs, df_test, gen_margin01, 20, device, model_margin01, threshold_margin01)
    toc = perf_counter()
    print(f"DataFrame creation for model margin 0.1: {((toc - tic)/60):.1f} min")
    
    model_margin05 = TripletLearner(base_channels=32)
    model_margin05.load_state_dict(torch.load("../models/in_article/margin05_600.pth",map_location=torch.device('cpu')))
    model_margin05 = model_margin05.to(device)
    model_margin05.eval()
    
    gen_margin05 = TripletGenerator(df_test, all_imgs, BATCH_TEST_SIZE, device, model_margin05, 0.5, return_id=True)
    threshold_margin05  = build_threshold(df_valid, all_imgs, device, model_margin05, 0.5, BATCH_VALID_SIZE, True)
    tic = perf_counter()
    df_fairness_margin05 = build_df_fairness(all_imgs, df_test, gen_margin05, 20, device, model_margin05, threshold_margin05)
    toc = perf_counter()
    print(f"DataFrame creation for model margin 0.5: {((toc - tic)/60):.1f} min")

    model_margin1 = TripletLearner(base_channels=32)
    model_margin1.load_state_dict(torch.load("../models/in_article/margin1_600.pth",map_location=torch.device('cpu')))
    model_margin1 = model_margin1.to(device)
    model_margin1.eval()
    
    gen_margin1 = TripletGenerator(df_test, all_imgs, BATCH_TEST_SIZE, device, model_margin1, 1., return_id=True)
    threshold_margin1  = build_threshold(df_valid, all_imgs, device, model_margin1, 1., BATCH_VALID_SIZE, True)
    tic = perf_counter()
    df_fairness_margin1 = build_df_fairness(all_imgs, df_test, gen_margin1, 20, device, model_margin1, threshold_margin1)
    toc = perf_counter()
    print(f"DataFrame creation for model margin 1: {((toc - tic)/60):.1f} min")
    
elif study == 'lr':
    model_lr3 = TripletLearner(base_channels=32)
    model_lr3.load_state_dict(torch.load("../models/in_article/lr3_600.pth",map_location=torch.device('cpu')))
    model_lr3 = model_lr3.to(device)
    model_lr3.eval()
    
    gen_lr3 = TripletGenerator(df_test, all_imgs, BATCH_TEST_SIZE, device, model_lr3, 0.2, return_id=True)
    threshold_lr3  = build_threshold(df_valid, all_imgs, device, model_lr3, 0.2, BATCH_VALID_SIZE, True)
    tic = perf_counter()
    df_fairness_lr3 = build_df_fairness(all_imgs, df_test, gen_lr3, 20, device, model_lr3, threshold_lr3)
    toc = perf_counter()
    print(f"DataFrame creation for model lr 10-3: {((toc - tic)/60):.1f} min")
    
    model_lr4 = TripletLearner(base_channels=32)
    model_lr4.load_state_dict(torch.load("../models/in_article/lr4_600.pth",map_location=torch.device('cpu')))
    model_lr4 = model_lr4.to(device)
    model_lr4.eval()
    
    gen_lr4 = TripletGenerator(df_test, all_imgs, BATCH_TEST_SIZE, device, model_lr4, 0.2, return_id=True)
    threshold_lr4  = build_threshold(df_valid, all_imgs, device, model_lr4, 0.2, BATCH_VALID_SIZE, True)
    tic = perf_counter()
    df_fairness_lr4 = build_df_fairness(all_imgs, df_test, gen_lr4, 20, device, model_lr4, threshold_lr4)
    toc = perf_counter()
    print(f"DataFrame creation for model lr 10-3: {((toc - tic)/60):.1f} min")
    
    model_lrschedule100 = TripletLearner(base_channels=32)
    model_lrschedule100.load_state_dict(torch.load("../models/in_article/lrschedule100_600.pth",map_location=torch.device('cpu')))
    model_lrschedule100 = model_lrschedule100.to(device)
    model_lrschedule100.eval()
    
    gen_lrschedule100 = TripletGenerator(df_test, all_imgs, BATCH_TEST_SIZE, device, model_lrschedule100, 0.2, return_id=True)
    threshold_lrschedule100  = build_threshold(df_valid, all_imgs, device, model_lrschedule100, 0.2, BATCH_VALID_SIZE, True)
    tic = perf_counter()
    df_fairness_lrschedule100 = build_df_fairness(all_imgs, df_test, gen_lrschedule100, 20, device, model_lrschedule100, threshold_lrschedule100)
    toc = perf_counter()
    print(f"DataFrame creation for model lr 10-3: {((toc - tic)/60):.1f} min")
    
    model_lrschedule300 = TripletLearner(base_channels=32)
    model_lrschedule300.load_state_dict(torch.load("../models/in_article/lrschedule300_600.pth",map_location=torch.device('cpu')))
    model_lrschedule300 = model_lrschedule300.to(device)
    model_lrschedule300.eval()
    
    gen_lrschedule300 = TripletGenerator(df_test, all_imgs, BATCH_TEST_SIZE, device, model_lrschedule300, 0.2, return_id=True)
    threshold_lrschedule300  = build_threshold(df_valid, all_imgs, device, model_lrschedule300, 0.2, BATCH_VALID_SIZE, True)
    tic = perf_counter()
    df_fairness_lrschedule300 = build_df_fairness(all_imgs, df_test, gen_lrschedule300, 20, device, model_lrschedule300, threshold_lrschedule300)
    toc = perf_counter()
    print(f"DataFrame creation for model lr 10-3: {((toc - tic)/60):.1f} min")
    
elif study == 'threshold':
    model_lowthrd = TripletLearner(base_channels=32)
    model_lowthrd.load_state_dict(torch.load("../models/in_article/base_600.pth",map_location=torch.device('cpu'))) #../models/in_article/base_model
    model_lowthrd = model_lowthrd.to(device)
    model_lowthrd.eval()

    gen_lowthrd = TripletGenerator(df_test, all_imgs, BATCH_TEST_SIZE, device, model_lowthrd, margin, return_id=True)
    threshold_lowthrd  = 0.48
    tic = perf_counter()
    df_fairness_lowthrd = build_df_fairness(all_imgs, df_test, gen_lowthrd, 20, device, model_lowthrd, threshold_lowthrd) #20
    toc = perf_counter()
    print(f"DataFrame creation for model low threshold: {((toc - tic)/60):.1f} min")
    
    model_highthrd = TripletLearner(base_channels=32)
    model_highthrd.load_state_dict(torch.load("../models/in_article/base_600.pth",map_location=torch.device('cpu'))) #../models/in_article/base_model
    model_highthrd = model_highthrd.to(device)
    model_highthrd.eval()

    gen_highthrd = TripletGenerator(df_test, all_imgs, BATCH_TEST_SIZE, device, model_highthrd, margin, return_id=True)
    threshold_highthrd  = 1.15
    tic = perf_counter()
    df_fairness_highthrd = build_df_fairness(all_imgs, df_test, gen_highthrd, 20, device, model_highthrd, threshold_highthrd) #20
    toc = perf_counter()
    print(f"DataFrame creation for model high threshold: {((toc - tic)/60):.1f} min")


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
        print(f"White Male - accuracy: {a} ({b}-{c})")
        b,a,c = np.round_(bootstrap_by_pairs(df_fairness[df_fairness['AB_WhiteMale']==1], \
            agg_func=lambda df: triplet_acc_fairness(df), num_bootstraps=1000, percentiles=[5,50,95]), 3)
        print(f"White Male - triplet accuracy: {a} ({b}-{c})")
        b,a,c = np.round_(bootstrap(df_fairness[df_fairness['AB_WhiteMale']==1], \
            agg_func=lambda df: df[(df['y_pred']==1)&(df['y_true']==0)]['id_A'].count()/df[df['y_pred']==1]['id_A'].count(), \
            num_bootstraps=1000, percentiles=[5,50,95]),3)
        print(f"White Male - FPR: {a} ({b}-{c})")
        b,a,c = np.round_(bootstrap(df_fairness[df_fairness['AB_WhiteMale']==1], \
            agg_func=lambda df: df[(df['y_pred']==0)&(df['y_true']==1)]['id_A'].count()/df[df['y_pred']==0]['id_A'].count(), \
            num_bootstraps=1000, percentiles=[5,50,95]),3)
        print(f"White Male - FNR: {a} ({b}-{c})")
        print("\n")
        b,a,c = np.round_(bootstrap(df_fairness[df_fairness['AB_NoWhiteMale']==1], \
            agg_func=lambda df: df['correct_predict'].mean(), num_bootstraps=1000, percentiles=[5,50,95]),3)
        print(f"Non-White Female - accuracy: {a} ({b}-{c})")
        b,a,c = np.round_(bootstrap_by_pairs(df_fairness[df_fairness['AB_NoWhiteMale']==1], \
            agg_func=lambda df: triplet_acc_fairness(df), num_bootstraps=1000, percentiles=[5,50,95]), 3)
        print(f"Non-White Female - triplet accuracy: {a} ({b}-{c})")
        b,a,c = np.round_(bootstrap(df_fairness[df_fairness['AB_NoWhiteMale']==1], \
            agg_func=lambda df: df[(df['y_pred']==1)&(df['y_true']==0)]['id_A'].count()/df[df['y_pred']==1]['id_A'].count(), \
            num_bootstraps=1000, percentiles=[5,50,95]),3)
        print(f"Non-White Female - FPR: {a} ({b}-{c})")
        b,a,c = np.round_(bootstrap(df_fairness[df_fairness['AB_NoWhiteMale']==1], \
            agg_func=lambda df: df[(df['y_pred']==0)&(df['y_true']==1)]['id_A'].count()/df[df['y_pred']==0]['id_A'].count(), \
            num_bootstraps=1000, percentiles=[5,50,95]),3)
        print(f"Non-White Female - FNR: {a} ({b}-{c})")
        print("\n")
        b,a,c = np.round_(bootstrap(df_fairness[(df_fairness['A_Black']==1)&(df_fairness['B_Black']==1)&(df_fairness['A_Male']==0)&(df_fairness['B_Male']==0)], \
            agg_func=lambda df: df['correct_predict'].mean(), num_bootstraps=1000, percentiles=[5,50,95]),3)
        print(f"Black Female - accuracy: {a} ({b}-{c})")
        b,a,c = np.round_(bootstrap_by_pairs(df_fairness[(df_fairness['A_Black']==1)&(df_fairness['B_Black']==1)&(df_fairness['A_Male']==0)&(df_fairness['B_Male']==0)], \
            agg_func=lambda df: triplet_acc_fairness(df), num_bootstraps=1000, percentiles=[5,50,95]), 3)
        print(f"Black Female - triplet accuracy: {a} ({b}-{c})")
        b,a,c = np.round_(bootstrap(df_fairness[(df_fairness['A_Black']==1)&(df_fairness['B_Black']==1)&(df_fairness['A_Male']==0)&(df_fairness['B_Male']==0)], \
            agg_func=lambda df: df[(df['y_pred']==1)&(df['y_true']==0)]['id_A'].count()/df[df['y_pred']==1]['id_A'].count(), \
            num_bootstraps=1000, percentiles=[5,50,95]),3)
        print(f"Black Female - FPR: {a} ({b}-{c})")
        b,a,c = np.round_(bootstrap(df_fairness[(df_fairness['A_Black']==1)&(df_fairness['B_Black']==1)&(df_fairness['A_Male']==0)&(df_fairness['B_Male']==0)], \
            agg_func=lambda df: df[(df['y_pred']==0)&(df['y_true']==1)]['id_A'].count()/df[df['y_pred']==0]['id_A'].count(), \
            num_bootstraps=1000, percentiles=[5,50,95]),3)
        print(f"Black Female - FNR: {a} ({b}-{c})")
        print("\n")

        
    else:
        print("Please select a correct analysis")

        
print("\n") 


print("------------------------------------------------")
print("GENDER ANALYSIS")
print("------------------------------------------------")
print("\n")

if study == 'sampling':
    print("Base sampling")
    results(df_fairness_base, "gender")
    print("Random sampling")
    results(df_fairness_randomsample, "gender")
    
elif study == 'normalization':
    print("Base")
    results(df_fairness_base, "gender")
    print("Normalized")
    results(df_fairness_normalized, "gender")
    
elif study == 'augmentation':
    print("Base augmentation")
    results(df_fairness_base, "gender")
    print("No augmentation")
    results(df_fairness_no_augment, "gender")
    print("High zoom")
    results(df_fairness_high_zoom, "gender")
    print("High deformation")
    results(df_fairness_deformation_double, "gender")
    print("High color jitter")
    results(df_fairness_jitter_double, "gender")
    print("High rotation")
    results(df_fairness_high_rotation, "gender")
    
elif study == 'depth':
    print("Depth 16")
    results(df_fairness_depth16, "gender")
    print("Depth 32")
    results(df_fairness_base, "gender")
    print("Depth 64")
    results(df_fairness_depth64, "gender")

elif study == 'margin':

    print("Margin 1")
    results(df_fairness_margin1, "gender")
    print("Margin 0.5")
    results(df_fairness_margin05, "gender")
    print("Margin 0.2")
    results(df_fairness_base, "gender")
    print("Margin 0.1")
    results(df_fairness_margin01, "gender")
    
elif study == 'lr':
    print("lr 5.10-4, scheduler starting at epoch 200")
    results(df_fairness_base, "gender")
    print("lr 10-3")
    results(df_fairness_lr3, "gender")
    print("lr 10-4")
    results(df_fairness_lr4, "gender")
    print("lr scheduler starting at epoch 100")
    results(df_fairness_lrschedule100, "gender")
    print("lr scheduler starting at epoch 300")
    results(df_fairness_lrschedule300, "gender")

elif study == 'threshold':
    print("Low threshold")
    results(df_fairness_lowthrd, "gender")
    print("Automatic threshold")
    results(df_fairness_base, "gender")
    print("High threshold")
    results(df_fairness_highthrd, "gender")

                                                
print("------------------------------------------------")
print("SKIN COLOR ANALYSIS")
print("------------------------------------------------")
print("\n")                                               

if study == 'sampling':
    print("Base sampling")
    results(df_fairness_base, "color")
    print("Random sampling")
    results(df_fairness_randomsample, "color")
    
elif study == 'normalization':
    print("Base")
    results(df_fairness_base, "color")
    print("Normalized")
    results(df_fairness_normalized, "color")
    
elif study == 'augmentation':
    print("Base augmentation")
    results(df_fairness_base, "color")
    print("No augmentation")
    results(df_fairness_no_augment, "color")
    print("High zoom")
    results(df_fairness_high_zoom, "color")
    print("High deformation")
    results(df_fairness_deformation_double, "color")
    print("High color jitter")
    results(df_fairness_jitter_double, "color")
    print("High rotation")
    results(df_fairness_high_rotation, "color")
    
elif study == 'depth':
    print("Depth 16")
    results(df_fairness_depth16, "color")
    print("Depth 32")
    results(df_fairness_base, "color")
    print("Depth 64")
    results(df_fairness_depth64, "color")

elif study == 'margin':
    print("Margin 1")
    results(df_fairness_margin1, "color")
    print("Margin 0.5")
    results(df_fairness_margin05, "color")
    print("Margin 0.2")
    results(df_fairness_base, "color")
    print("Margin 0.1")
    results(df_fairness_margin01, "color")
    
elif study == 'lr':
    print("lr 5.10-4, scheduler starting at epoch 200")
    results(df_fairness_base, "color")
    print("lr 10-3")
    results(df_fairness_lr3, "color")
    print("lr 10-4")
    results(df_fairness_lr4, "color")
    print("lr scheduler starting at epoch 100")
    results(df_fairness_lrschedule100, "color")
    print("lr scheduler starting at epoch 300")
    results(df_fairness_lrschedule300, "color")

elif study == 'threshold':
    print("Low threshold")
    results(df_fairness_lowthrd, "color")
    print("Automatic threshold")
    results(df_fairness_base, "color")
    print("High threshold")
    results(df_fairness_highthrd, "color")


print("------------------------------------------------")
print("INTERSECTIONAL ANALYSIS")
print("------------------------------------------------")
print("\n")

if study == 'sampling':
    print("Base sampling")
    results(df_fairness_base, "intersectional")
    print("Random sampling")
    results(df_fairness_randomsample, "intersectional")
    
elif study == 'normalization':
    print("Base")
    results(df_fairness_base, "intersectional")
    print("Normalized")
    results(df_fairness_normalized, "intersectional")
    
elif study == 'augmentation':
    print("Base augmentation")
    results(df_fairness_base, "intersectional")
    print("No augmentation")
    results(df_fairness_no_augment, "intersectional")
    print("High zoom")
    results(df_fairness_high_zoom, "intersectional")
    print("High deformation")
    results(df_fairness_deformation_double, "intersectional")
    print("High color jitter")
    results(df_fairness_jitter_double, "intersectional")
    print("High rotation")
    results(df_fairness_high_rotation, "intersectional")
    
elif study == 'depth':
    print("Depth 16")
    results(df_fairness_depth16, "intersectional")
    print("Depth 32")
    results(df_fairness_base, "intersectional")
    print("Depth 64")
    results(df_fairness_depth64, "intersectional")
                                                
elif study == 'margin':
    print("Margin 1")
    results(df_fairness_margin1, "intersectional")
    print("Margin 0.5")
    results(df_fairness_margin05, "intersectional")
    print("Margin 0.2")
    results(df_fairness_base, "intersectional")
    print("Margin 0.1")
    results(df_fairness_margin01, "intersectional")
    
elif study == 'lr':
    print("lr 5.10-4, scheduler starting at epoch 200")
    results(df_fairness_base, "intersectional")
    print("lr 10-3")
    results(df_fairness_lr3, "intersectional")
    print("lr 10-4")
    results(df_fairness_lr4, "intersectional")
    print("lr scheduler starting at epoch 100")
    results(df_fairness_lrschedule100, "intersectional")
    print("lr scheduler starting at epoch 300")
    results(df_fairness_lrschedule300, "intersectional")

elif study == 'threshold':
    print("Low threshold")
    results(df_fairness_lowthrd, "intersectional")
    print("Automatic threshold")
    results(df_fairness_base, "intersectional")
    print("High threshold")
    results(df_fairness_highthrd, "intersectional")


print("\n")
torch.cuda.empty_cache()
print("Cleared cache")