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

seed = 121
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


# SOME USEFUL FUNCTIONS
###############################################################################

PATH = "lfw/lfw-deepfunneled/"

ATTR_FILE = 'lfw_att_73.mat'

def from_tensor_to_numpy(t):
    """Transforms a torch tensor to a numpy array.
        input: t a tensor of shape (1,3,60,60) or (3,60,60)
        output: n a numpy array of shape (60,60,3)"""
    if len(t.shape)>3 :
        t = t.detach().clone()[0]
    else :
        t = t.detach().clone()
    n = t.numpy()
    n = n.transpose((1, 2, 0))
    return n

def from_numpy_to_tensor(n):
    """Transforms a numpy array to a torch tensor.
        input: n a numpy array of shape (60,60,3)
        output: t a tensor of shape (1,3,60,60)"""
    image_transforms = transforms.Compose([transforms.ToTensor(),])
    return image_transforms(n)[None,:]

def rewrite_names(name_list):
    """Useful function for rewriting paths, replacing '\\' with '/'"""
    new_list = []
    for stri in name_list:
        new_list.append(re.sub('[\\\\]', '/', stri))
    return(new_list)

def resize_and_crop(img):
    """Resize (100,100) and crop the image (60,60)"""
    return resize(
        img, (100, 100),
        preserve_range=True, mode='reflect', anti_aliasing=True
    )[20:80,20:80,:]

def open_one_image_tensor(path, PATH=PATH):
    """Open an image from its path and return the image in form of a tensor.
    Image values are in (0,1)."""
    image_transforms = transforms.Compose([transforms.ToTensor(),])
    return (image_transforms(
            resize_and_crop(imread(PATH+path).astype("float32"))).unsqueeze(0))

def open_one_image_numpy(path, PATH=PATH):
    """Open an image from its path and return the image in form of an array.
    Image values are in (0,1)."""
    return (resize_and_crop(imread(PATH+path).astype("float32")))

def open_all_images(id_to_path, PATH=PATH):
    """Open all images from the file
        input: a dictionnary with keys=id of the images,
        values=path of the images
        output: tensor of shape (nb img, 3, 60, 60)"""
    all_imgs = []
    for _,path in id_to_path.items():
        all_imgs += [open_one_image_tensor(path, PATH=PATH)]
    return torch.vstack(all_imgs)


# BUILD DATAFRAME
###############################################################################

def create_dataframe(PATH=PATH):
    """Create the dataframe.
        output: df the dataframe created
                (with columns id, Classids, Names, Paths)
                all_imgs a tensor of shape (nb img, 3, 60, 60)"""

    dirs = sorted(os.listdir(PATH))

    classids = pd.Series([
        classid for classid, name in enumerate(dirs)
        for _ in sorted(os.listdir(PATH+name))
    ])
    names = pd.Series([
        name for _, name in enumerate(dirs)
        for _ in sorted(os.listdir(PATH+name))
    ])
    img_paths = pd.Series([
        name + '/' + img for _, name in enumerate(dirs)
        for img in sorted(os.listdir(PATH+name))
    ])
    nb_imgs = len(classids)
    nb_indiv = len(classids.unique())

    print("Number of individuals: ", nb_indiv)
    print("Number of total images: ", nb_imgs)

    df = pd.DataFrame({"Classid":classids, "Name":names, "Path":img_paths})

    path_to_id = {path:img_idx for img_idx,path in enumerate(img_paths)}
    id_to_path = {img_idx:path for path,img_idx in path_to_id.items()}

    all_imgs = open_all_images(id_to_path, PATH=PATH)

    print("images weigh ",
        str(round(all_imgs.element_size() * all_imgs.nelement() / 1e9, 2)),
        "GB")

    return df, all_imgs

def extend_dataframe(df):
    """Extend the dataframe.
        output: df_final the dataframe extended with all the attributes
                (with columns id, Classids, Names, Paths as df +
                all the attributes columns)
                Where the attributes are not defined, NaN appear"""

    data_dict = mat73.loadmat(ATTR_FILE)
    data_dict['name'] = rewrite_names(data_dict['name'])

    path_to_label = {
        path:label for path,label in zip(data_dict['name'], data_dict['label'])
    }

    df2 = pd.DataFrame.from_dict(
        path_to_label, orient='index', columns=data_dict['AttrName']
    )

    df_final = df.join(df2, on='Path')

    return df_final


# BUILD PAIRS FOR RANDOM SAMPLING
###############################################################################

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