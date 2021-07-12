import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from builder import open_one_image_numpy, open_one_image_tensor, from_tensor_to_numpy #, unnormalize_img #unnormalize_img, unnormalize_img_from_path


# USEFUL FUNCTIONS
#################################################################################

# MEAN = np.array([131.60774134, 105.40847531,  88.49154859])

PATH = "lfw/lfw-deepfunneled/"

# PRINT FROM DATA
#################################################################################

def print_img(x):
    plt.imshow(x)#unnormalize_img_from_path(mean, path))
    plt.axis('off')
    plt.show()

def print_img_from_path(path):
    x = open_one_image_numpy(path)/255
    # x = open_one_image_tensor
    plt.imshow(x)#unnormalize_img_from_path(mean, path))
    plt.axis('off')
    plt.show()


def print_img_from_id(df, id):
    path = df.Path.iloc[id].values[0]
    x = open_one_image_numpy(path)/255
    plt.imshow(x) #unnormalize_img_from_path(mean, path))
    plt.axis('off')
    plt.show()


def print_img_from_classid(df, classid):
    if df.Classid.isin([classid]).any().any():
        list_paths = df.Path[df.Classid==classid].values
        fig,ax = plt.subplots(1,len(list_paths))
        for i in range(len(list_paths)):
            plt.subplot(1,len(list_paths),i+1)
            x = open_one_image_numpy(list_paths[i])/255
            plt.imshow(x)#unnormalize_img_from_path(mean, list_paths[i]))
            plt.axis('off')
        plt.show()
    else:
        print("This class is not available")

# def print_img_category(attribute):
#     fig,ax = plt.subplots(3,4,figsize=(16, 9))
#     for i in range(12):
#         plt.subplot(3,4,i+1)
#         plt.imshow(resize100(imread(PATH+df[df[attribute]==1]['Path'].sample().iloc[0]))/255)
#         plt.axis('off')
#     plt.show()

def print_from_gen(gen,idx):
    xa, xp, xn = gen[idx]
    # xa = unnormalize_img(mean, xa)
    # xp = unnormalize_img(mean, xp)
    # xn = unnormalize_img(mean, xn)
    # xa = xa.numpy()
    # xp = xp.numpy()
    # xn = xn.numpy()
    plt.figure(figsize=(16, 9))

    for i in range(5):
        x = from_tensor_to_numpy(xa[i])#*255)#xa[i].transpose((1, 2, 0))
        plt.subplot(3, 5, i + 1)
        plt.title("anchor")
        plt.imshow(x)#unnormalize_img(mean,x))     #(x + MEAN)/255)
        plt.axis('off')

    for i in range(5):
        x = from_tensor_to_numpy(xp[i])#*255)#xp[i].transpose((1, 2, 0))
        plt.subplot(3, 5, i + 6)
        plt.title("positive")
        plt.imshow(x)
        plt.axis('off')

    for i in range(5):
        x = from_tensor_to_numpy(xn[i])#*255)#xn[i].transpose((1, 2, 0))
        plt.subplot(3, 5, i + 11)
        plt.title("negative")
        plt.imshow(x)
        plt.axis('off')

    plt.show()



# GROUP PRINT
#################################################################################

def print_pairs(df, xa, xp, id_x):
    list_paths = [df.Path.iloc[xa[id_x]], df.Path.iloc[xp[id_x]]]
    fig,ax = plt.subplots(1,2)
    plt.subplot(1,2,1)
    plt.imshow(open_one_image_numpy(list_paths[0])) #unnormalize_img_from_path(mean,list_paths[0]))
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(open_one_image_numpy(list_paths[1])) #unnormalize_img_from_path(mean,list_paths[1]))
    plt.axis('off')
    plt.show()


# HISTOGRAMS
#################################################################################

def print_hist(pos_loss, neg_loss):
    fig,ax = plt.subplots(1,1,figsize=(6,3),dpi=100,num=1)
    bins=np.linspace(0.,1,30)
    plt.hist(pos_loss,bins=bins,label='positive loss', alpha=0.3)
    plt.hist(neg_loss,bins=bins,label='negative loss', alpha=0.3)
    # plt.xlim([0.,1.])
    plt.legend()
    plt.show()