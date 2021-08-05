import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from builder import open_one_image_numpy, from_tensor_to_numpy


# USEFUL FUNCTIONS
###############################################################################

PATH = "lfw/lfw-deepfunneled/"

# PRINT FROM DATA
###############################################################################

def print_img(x):
    """Print the image x directly.
        For the image to correctly display, x must be a numpy array
        whose values are in (0,1)"""
    plt.imshow(x)
    plt.axis('off')
    plt.show()

def print_img_from_path(path):
    """Print the image from its path"""
    x = open_one_image_numpy(path)/255
    plt.imshow(x)
    plt.axis('off')
    plt.show()


def print_img_from_id(df, id):
    """Print the image from its id in df"""
    path = df.Path.iloc[id].values[0]
    x = open_one_image_numpy(path)/255
    plt.imshow(x)
    plt.axis('off')
    plt.show()


def print_img_from_classid(df, classid):
    """Print the images corresponding to a certain classid"""
    if df.Classid.isin([classid]).any().any():
        list_paths = df.Path[df.Classid==classid].values
        fig,ax = plt.subplots(1,len(list_paths))
        for i in range(len(list_paths)):
            plt.subplot(1,len(list_paths),i+1)
            x = open_one_image_numpy(list_paths[i])/255
            plt.imshow(x)
            plt.axis('off')
        plt.show()
    else:
        print("This class is not available")

# DO NOT WORK
def print_img_category(all_imgs, df, attribute):
    fig,ax = plt.subplots(3,4,figsize=(16, 9))
    idx = df[df[attribute]==1.].sample(12).index
    for i in range(12):
        plt.subplot(3,4,i+1)
        img = from_tensor_to_numpy(all_imgs[idx[i]])/255
        plt.imshow(img)
        plt.axis('off')
    plt.show()

def print_from_gen(gen,idx):
    """Print some images from a generator gen.
    	The function displays the first 5 images in the batch idx"""

    xa, xp, xn = gen[idx]
    plt.figure(figsize=(16, 9))

    for i in range(5):
        x = from_tensor_to_numpy(xa[i])/255
        plt.subplot(3, 5, i + 1)
        plt.title("anchor")
        plt.imshow(x)
        plt.axis('off')

    for i in range(5):
        x = from_tensor_to_numpy(xp[i])/255
        plt.subplot(3, 5, i + 6)
        plt.title("positive")
        plt.imshow(x)
        plt.axis('off')

    for i in range(5):
        x = from_tensor_to_numpy(xn[i])/255
        plt.subplot(3, 5, i + 11)
        plt.title("negative")
        plt.imshow(x)
        plt.axis('off')

    plt.show()

def print_from_gen2(gen):

    x, _, _ = gen[0]

    plt.figure(figsize=(10,10*len(x)*len(gen)))

    for k in range(len(gen)+1):

        xa, xp, xn = gen[k]

        for i in range(len(xa)):
        	
            anchor_numpy = from_tensor_to_numpy(xa[i])/255
            positive_numpy = from_tensor_to_numpy(xp[i])/255
            negative_numpy = from_tensor_to_numpy(xn[i])/255

            plt.subplot((len(gen)+1)*len(xa), 3, 3*len(xa)*k+3*i+1)
#             plt.title("anchor")
            plt.imshow(anchor_numpy)
            plt.axis('off')

            plt.subplot((len(gen)+1)*len(xa), 3, 3*len(xa)*k+3*i+2)
#             plt.title("positive")
            plt.imshow(positive_numpy)
            plt.axis('off')

            plt.subplot((len(gen)+1)*len(xa), 3, 3*len(xa)*k+3*i+3)
#             plt.title("negative")
            plt.imshow(negative_numpy)
            plt.axis('off')

    plt.show()

# DO NOT WORK
def print_from_loader(loader):

    plt.figure(figsize=(16, 9))

    for step, (anchor_img, positive_img, negative_img) \
    in enumerate(tqdm(loader, desc="Processing", leave=False)):

        anchor_numpy = from_tensor_to_numpy(anchor_img)/255
        plt.subplot(len(loader), 3, 1)
        plt.title("anchor")
        plt.imshow(anchor_numpy)
        plt.axis('off')

        positive_numpy = from_tensor_to_numpy(positive_img)/255
        plt.subplot(len(loader), 3, 2)
        plt.title("positive")
        plt.imshow(positive_numpy)
        plt.axis('off')

        negative_numpy = from_tensor_to_numpy(negative_img)/255
        plt.subplot(len(loader), 3, 3)
        plt.title("negative")
        plt.imshow(negative_numpy)
        plt.axis('off')

    plt.show()



# GROUP PRINT
###############################################################################

def print_pair(img1, img2):
    """Print a pair of two images"""
    fig,ax = plt.subplots(1,2)
    plt.subplot(1,2,1)
    plt.imshow(from_tensor_to_numpy(img1)/255)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(from_tensor_to_numpy(img2)/255)
    plt.axis('off')
    plt.show()


# HISTOGRAMS
###############################################################################

def print_hist_dist(pos_dist, neg_dist):
    """Print the distance histogram"""
    fig,ax = plt.subplots(1,1,figsize=(6,3),dpi=100,num=1)
    bins=np.linspace(0.,10.,50)
    plt.title('Histogram of distances')
    plt.hist(pos_dist,bins=bins,
        label='distance between anchor and positive image', alpha=0.3)
    plt.hist(neg_dist,bins=bins,
        label='distance between anchor and negative image', alpha=0.3)
    # plt.xlim([0.,1.])
    plt.legend()
    plt.show()

def print_hist_dist_zoom(pos_dist, neg_dist):
    """Print the distance histogram with a zoom on the 0-2  x window"""
    fig,ax = plt.subplots(1,1,figsize=(6,3),dpi=100,num=1)
    bins=np.linspace(0.,2.,50)
    plt.title('Histogram of distances')
    plt.hist(pos_dist,bins=bins,
        label='distance between anchor and positive image', alpha=0.3)
    plt.hist(neg_dist,bins=bins,
        label='distance between anchor and negative image', alpha=0.3)
    plt.xlim([0.,2.])
    plt.legend()
    plt.show()

def print_hist_loss(loss):
    """Print the loss histogram"""
    fig,ax = plt.subplots(1,1,figsize=(6,3),dpi=100,num=1)
    bins=np.linspace(0.,5.,30)
    plt.title('Histogram of the loss')
    plt.hist(loss,bins=50,label='Triplet Loss')
    # plt.xlim([0.,1.])
    plt.legend()
    plt.show()