import numpy as np
import pandas as pd
import mat73
import random
import itertools
import re
from skimage.io import imread
from skimage.transform import resize
import os


# SOME USEFUL FUNCTIONS
##################################################################################

PATH = "lfw/lfw-deepfunneled/"

def rewrite_names(name_list):
    new_list = []
    for stri in name_list:
        new_list.append(re.sub('[\\\\]', '/', stri))
    return(new_list)

def resize100(img):
    return resize(img, (100, 100), preserve_range=True, mode='reflect', anti_aliasing=True)[20:80,20:80,:]

def open_all_images(id_to_path):
    all_imgs = []
    for path in id_to_path.values():
        all_imgs += [np.expand_dims(resize100(imread(PATH+path)),0)]
    return np.vstack(all_imgs)


# BUILD DATAFRAME
##################################################################################

def create_dataframe():

    dirs = sorted(os.listdir(PATH))
    name_to_classid = {d:i for i,d in enumerate(dirs)}
    classid_to_name = {v:k for k,v in name_to_classid.items()}
    num_classes = len(name_to_classid)
    # print("number of total classes: "+str(num_classes))

    # read all directories
    img_paths = {c:[directory + "/" + img for img in sorted(os.listdir(PATH+directory))] 
                for directory,c in name_to_classid.items()}

    # retrieve all images
    all_images_path = []
    for img_list in img_paths.values():
        all_images_path += img_list

    # map to integers
    path_to_id = {v:k for k,v in enumerate(all_images_path)}
    id_to_path = {v:k for k,v in path_to_id.items()}

    # build mappings between images and class
    classid_to_ids = {k:[path_to_id[path] for path in v] for k,v in img_paths.items()}
    id_to_classid = {v:c for c,imgs in classid_to_ids.items() for v in imgs}

    classid_to_path = img_paths
    path_to_classid = {}
    for key, value in classid_to_path.items():
        for string in value:
            path_to_classid.setdefault(string, key)

    all_imgs = open_all_images(id_to_path)
    mean = np.mean(all_imgs, axis=(0,1,2))
    all_imgs -= mean
    print("images weigh ", str(round(all_imgs.nbytes / 1e9,2)), "GB")

    data_dict = mat73.loadmat('lfw_att_73.mat')
    data_dict.name = rewrite_names(data_dict.name)

    attributes = ['Path'] + data_dict.AttrName #['Id'] + ['Classid'] + ['Name'] + ['Img'] +

    values = np.zeros((len(data_dict.name),len(data_dict.AttrName)+1),dtype=object)

    for i in range(len(data_dict.name)):
        values[i] = [data_dict.name[i]] + list(data_dict.label[i])

    df = pd.DataFrame(values, columns=attributes)

    df.insert(0,'Id',int)
    for id,path in id_to_path.items():
        df['Id'][df['Path']==path]=id

    df.insert(1,'Classid',int)
    for path,classid in path_to_classid.items():
        df['Classid'][df['Path']==path]=int(classid)

    df.insert(2,'Name',str)
    for name,classid in name_to_classid.items():
        df['Name'][df['Classid']==classid]=name

    df.insert(3,'Img',np.array)
    for i in range(len(df.Img)):
        df['Img'].iloc[i]=all_imgs[df['Id'].iloc[i]]

    print("number of classes: "+str(len(df.Classid.unique())))

    return df


# BUILD PAIRS
##################################################################################

def build_pos_pairs_for_id(df, classid, max_num=50):
    id_imgs = df.index[df.Classid==classid]
    if len(id_imgs) <= 1:
        return []
    
    pos_pairs = list(itertools.combinations(id_imgs, 2))
    
    random.shuffle(pos_pairs)
    return pos_pairs[:max_num]


def build_positive_pairs(df, class_id_range):
    listX1 = []
    listX2 = []
    
    for class_id in class_id_range:
        pos = build_pos_pairs_for_id(df, class_id)
        for pair in pos:
            listX1 += [pair[0]]
            listX2 += [pair[1]]
            
    perm = np.random.permutation(len(listX1))
    return np.array(listX1)[perm], np.array(listX2)[perm]


def build_negative_pairs(df, classid_range, pairs_num=128):
    listX1 = []
    listX2 = []
    classid_range = [i for i in classid_range if df.Classid.isin([i]).any().any()]
    for i in range(pairs_num):
        list_classids = random.sample(classid_range, 2)
        id1 = df.index[df.Classid==list_classids[0]][random.randint(0,len(df.index[df.Classid==list_classids[0]]))-1]
        id2 = df.index[df.Classid==list_classids[1]][random.randint(0,len(df.index[df.Classid==list_classids[1]]))-1]
        listX1.append(id1)
        listX2.append(id2)
        
    perm = np.random.permutation(len(listX1))
    return np.array(listX1)[perm], np.array(listX2)[perm]


