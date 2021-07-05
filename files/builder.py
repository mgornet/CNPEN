import numpy as np
import pandas as pd
import mat73
import random
import itertools
import re
from skimage.io import imread
from skimage.transform import resize
import os
import torch
from torchvision import transforms


# SOME USEFUL FUNCTIONS
##################################################################################

PATH = "lfw/lfw-deepfunneled/"

ATTR_FILE = 'lfw_att_73.mat'

def rewrite_names(name_list):
    new_list = []
    for stri in name_list:
        new_list.append(re.sub('[\\\\]', '/', stri))
    return(new_list)

def resize100(img):
    return resize(img, (100, 100), preserve_range=True, mode='reflect', anti_aliasing=True)[20:80,20:80,:]

def open_all_images(id_to_path):
    all_imgs = []
    image_transforms = transforms.Compose([transforms.ToTensor(),])
    for _,path in id_to_path.items():
        all_imgs += [image_transforms(resize100(imread(PATH+path))).unsqueeze(0)]
    return torch.vstack(all_imgs)


# BUILD DATAFRAME
##################################################################################

def create_dataframe():

    dirs = sorted(os.listdir(PATH))

    classids = pd.Series([classid for classid, name in enumerate(dirs) for _ in sorted(os.listdir(PATH+name))])
    names = pd.Series([name for _, name in enumerate(dirs) for _ in sorted(os.listdir(PATH+name))])
    img_paths = pd.Series([name + '/' + img for _, name in enumerate(dirs) for img in sorted(os.listdir(PATH+name))])
    nb_imgs = len(classids)
    nb_indiv = len(classids.unique())

    print("Number of individuals: ", nb_indiv)
    print("Number of total images: ", nb_imgs)

    df = pd.DataFrame({"Classid":classids, "Name":names, "Path":img_paths})

    path_to_id = {v:k for k,v in enumerate(img_paths)}
    id_to_path = {v:k for k,v in path_to_id.items()}

    all_imgs = open_all_images(id_to_path)
    mean = torch.mean(all_imgs, axis=(0,1,2))
    all_imgs -= mean
    print("images weigh ", str(round(all_imgs.element_size() * all_imgs.nelement() / 1e9,2)), "GB")

    return df, all_imgs, mean

# MARCHE PAS
def extend_dataframe(df):

    data_dict = mat73.loadmat(ATTR_FILE)
    data_dict.name = rewrite_names(data_dict.name)

    for attr in data_dict.AttrName:
        df[attr]=np.nan

    path_to_label = {path:label for path in data_dict.name for label in data_dict.label}

    for path, label in path_to_label.items():
        for i, attr in zip(range(len(data_dict.AttrName)),data_dict.AttrName):
            df[attr][df.Path==path]=int(label[i])

    return df


# BUILD PAIRS
##################################################################################

# def build_pos_pairs_for_id(df, classid, max_num=50):
#     id_imgs = df.index[df.Classid==classid]
#     if len(id_imgs) <= 1:
#         return []
    
#     pos_pairs = list(itertools.combinations(id_imgs, 2))
    
#     random.shuffle(pos_pairs)
#     return pos_pairs[:max_num]


# def build_positive_pairs(df, class_id_range):
#     listX1 = []
#     listX2 = []
    
#     for class_id in class_id_range:
#         pos = build_pos_pairs_for_id(df, class_id)
#         for pair in pos:
#             listX1 += [pair[0]]
#             listX2 += [pair[1]]
            
#     perm = np.random.permutation(len(listX1))
#     return np.array(listX1)[perm], np.array(listX2)[perm]


# def build_negative_pairs(df, classid_range, pairs_num=128):
#     listX1 = []
#     listX2 = []
#     classid_range = [i for i in classid_range if df.Classid.isin([i]).any().any()]
#     for i in range(pairs_num):
#         list_classids = random.sample(classid_range, 2)
#         id1 = df.index[df.Classid==list_classids[0]][random.randint(0,len(df.index[df.Classid==list_classids[0]]))-1]
#         id2 = df.index[df.Classid==list_classids[1]][random.randint(0,len(df.index[df.Classid==list_classids[1]]))-1]
#         listX1.append(id1)
#         listX2.append(id2)
        
#     perm = np.random.permutation(len(listX1))
#     return np.array(listX1)[perm], np.array(listX2)[perm]