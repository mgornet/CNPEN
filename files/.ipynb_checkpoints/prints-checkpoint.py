import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from builder import open_one_image_numpy, from_tensor_to_numpy

seed = 121
np.random.seed(seed)


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

# GROUP PRINT
###############################################################################

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

def print_img_category(all_imgs, df, attribute, inverse=False):
    fig,ax = plt.subplots(3,4,figsize=(16, 9))
    if inverse :
        idx = df[df[attribute]==0].sample(12).index
    else :
        idx = df[df[attribute]==1.].sample(12).index
    for i in range(12):
        plt.subplot(3,4,i+1)
        img = from_tensor_to_numpy(all_imgs[idx[i]]/255)
        plt.imshow(img)
        plt.axis('off')
    plt.show()

def print_from_gen(gen,idx):
    """Print some images from a generator gen.
        The function displays the first 5 images in the batch idx"""

    xa, xp, xn = gen[idx]
    plt.figure(figsize=(16, 9))

    for i in range(5):
        x = from_tensor_to_numpy(xa[i]/255)
        plt.subplot(3, 5, i + 1)
        plt.title("anchor")
        plt.imshow(x)
        plt.axis('off')

    for i in range(5):
        x = from_tensor_to_numpy(xp[i]/255)
        plt.subplot(3, 5, i + 6)
        plt.title("positive")
        plt.imshow(x)
        plt.axis('off')

    for i in range(5):
        x = from_tensor_to_numpy(xn[i]/255)
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

            anchor_numpy = from_tensor_to_numpy(xa[i]/255)
            positive_numpy = from_tensor_to_numpy(xp[i]/255)
            negative_numpy = from_tensor_to_numpy(xn[i]/255)

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
        label='Anchor and Positive', alpha=0.3)
    plt.hist(neg_dist,bins=bins,
        label='Anchor and Negative', alpha=0.3)
    # plt.xlim([0.,1.])
    ax.set_xlabel('Distance')
    ax.set_ylabel('Number of instances')
    plt.legend()
    plt.show()

def print_hist_dist_zoom(pos_dist, neg_dist, zoom=2.):
    """Print the loss histogram"""
    maxi=zoom
    fig,ax = plt.subplots(1,1,figsize=(6,3),dpi=100,num=1)
    bins=np.linspace(0.,maxi,70)
    plt.title('Histogram of distances')
    plt.hist(pos_dist,bins=bins,
             label='Anchor and Positive', alpha=0.3)
    plt.hist(neg_dist,bins=bins,
             label='Anchor and Negative', alpha=0.3)
    plt.xlim([0.,maxi])
    ax.set_xlabel('Distance')
    ax.set_ylabel('Number of instances')
    plt.legend()
    plt.show()

def print_hist_loss(loss):
    """Print the loss histogram"""
    fig,ax = plt.subplots(1,1,figsize=(6,3),dpi=100,num=1)
    bins=np.linspace(0.,5.,30)
    plt.title('Histogram of the loss')
    plt.hist(loss,bins=50,label='Triplet Loss')
    ax.set_xlabel('Loss')
    ax.set_ylabel('Number of instances')
    plt.legend()
    plt.show()

# PLOT FUNCTIONS
###############################################################################

def print_roc(fpr, tpr, roc_auc):
    plt.figure()
    lw=2
    plt.plot(fpr, tpr, color="darkorange", lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0,1],[0,1], color="navy",lw=lw, linestyle="--")
    plt.xlim([0.,1.])
    plt.ylim([0.,1.])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()

def print_prec_recall(precision, recall, auc):
    plt.figure()
    lw=2
    plt.plot(recall, precision, color="darkorange",
        lw=lw, label="Precision-Recall curve (auc = %0.2f)" % auc)
    plt.axhline(y=0.5, xmin=0.0, xmax=1., color="navy", linestyle="--", lw=lw)
    plt.ylim([0.5,1.])
    plt.xlim([0.,1.])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.legend(loc="lower right")
    plt.show()

def print_logistic_regression(pos_dist, neg_dist,
    y_pred_proba_logistic, threshold, maxi=2.):
    """Print the curves from logistic regression above the histogram
    and show threshold"""
    X = pos_dist + neg_dist
    fig,ax1 = plt.subplots(1,1,figsize=(10,5),dpi=100,num=1)
    bins=np.linspace(0.,maxi,70)
    plt.title('Histogram of distances and logistic regression')
    ax1.hist(pos_dist,bins=bins,
             label='Anchor and Positive', alpha=0.3)
    ax1.hist(neg_dist,bins=bins,
             label='Anchor and Negative', alpha=0.3)
    plt.xlim([0.,maxi])
    ax1.set_xlabel('Distance')
    ax1.set_ylabel('Number of instances')
    ax2 = ax1.twinx()
    ax2.plot(X,y_pred_proba_logistic[:,0],'.',
        color='orange',label='proba y=0')
    ax2.plot(X,y_pred_proba_logistic[:,1],'.',
        color='green', label='proba y=1')
    ax2.set_ylabel('Probability')
    ax2.vlines(threshold, ymin=0, ymax=1, color='red', linestyles='--',
               lw=2, label=f'Threshold: {round(threshold,2)}')
    ax1.legend(loc='center right')
    ax2.legend(loc='center left')
    plt.ylim([0.,1.])
    plt.show()