import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize


# USEFUL FUNCTIONS
#################################################################################

MEAN = np.array([131.60774134, 105.40847531,  88.49154859])

PATH = "lfw/lfw-deepfunneled/"

def resize100(img):
    return resize(img, (100, 100), preserve_range=True, mode='reflect', anti_aliasing=True)[20:80,20:80,:]


# PRINT FROM DATA
#################################################################################

def print_img_from_path(path):
    plt.imshow(resize100(imread(PATH+path))/255)
    plt.axis('off')
    plt.show()


def print_img_from_id(df, id):
    path = df.Path[df.index==id].values[0]
    plt.imshow(resize100(imread(PATH+path))/255)
    plt.axis('off')
    plt.show()


def print_img_from_classid(df, classid):
    if df.Classid.isin([classid]).any().any():
        list_paths = df.Path[df.Classid==classid].values
        fig,ax = plt.subplots(1,len(list_paths))
        for i in range(len(list_paths)):
            plt.subplot(1,len(list_paths),i+1)
            plt.imshow(resize100(imread(PATH+list_paths[i])/255))
            plt.axis('off')
        plt.show()
    else:
        print("This class is not available")

def print_img_category(attribute):
    fig,ax = plt.subplots(3,4,figsize=(16, 9))
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.imshow(resize100(imread(PATH+df[df[attribute]==1]['Path'].sample().iloc[0]))/255)
        plt.axis('off')
    plt.show()

def print_from_gen(gen,idx):
    xa, xp, xn = gen[idx]
    xa = xa.numpy()
    xp = xp.numpy()
    xn = xn.numpy()
    plt.figure(figsize=(16, 9))

    for i in range(5):
        x = xa[i].transpose((1, 2, 0))
        plt.subplot(3, 5, i + 1)
        plt.title("anchor")
        plt.imshow((x + MEAN)/255)
        plt.axis('off')

    for i in range(5):
        x = xp[i].transpose((1, 2, 0))
        plt.subplot(3, 5, i + 6)
        plt.title("positive")
        plt.imshow((x + MEAN) / 255)
        plt.axis('off')

    for i in range(5):
        x = xn[i].transpose((1, 2, 0))
        plt.subplot(3, 5, i + 11)
        plt.title("negative")
        plt.imshow((x + MEAN) / 255)
        plt.axis('off')

    plt.show()



# GROUP PRINT
#################################################################################

def print_pairs(df, xa, xp, id_x):
    list_paths = [df.Path.iloc[xa[id_x]], df.Path.iloc[xp[id_x]]]
    fig,ax = plt.subplots(1,2)
    plt.subplot(1,2,1)
    plt.imshow(resize100(imread(PATH+list_paths[0])/255))
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(resize100(imread(PATH+list_paths[1])/255))
    plt.axis('off')
    plt.show()


# HISTOGRAMS
#################################################################################

def print_hist(pos_loss, neg_loss):
    fig,ax = plt.subplots(1,1,figsize=(6,3),dpi=100,num=1)
    plt.hist(pos_loss,bins=10,label='positive loss')
    plt.hist(neg_loss,bins=10,label='negative loss')
    plt.legend()
    plt.show()