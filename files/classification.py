import numpy as np
from tqdm.notebook import tqdm
import time
import torch
from triplet import distance


# PATH = "./CNPEN/files/"

# TESTING LOOP
###############################################################################

def authentification_img(img1, img2, device, model, threshold, verbose=False):

    img1 = img1[None,:,:,:].to(device)
    img2 = img2[None,:,:,:].to(device)

    img1_out = model(img1)
    img2_out = model(img2)

    dist = distance(img1_out, img2_out).cpu().detach().item()
    
    if verbose == True :
        print("Distance: ", round(dist,2))

    if dist < threshold :
        return 1

    return 0

def predict(loader, device, model, threshold):

    pred_pos = []
    pred_neg = []
    
    for step, (anchor_img, positive_img, negative_img) in enumerate(loader) :
        for i in range(len(anchor_img)):
            
            bin_pos = authentification_img(
                anchor_img[i], positive_img[i], device, model, threshold
            )
            pred_pos.append(bin_pos)
            
            bin_neg = authentification_img(
                anchor_img[i], negative_img[i], device, model, threshold
            )
            pred_neg.append(bin_neg)
    
    return pred_pos, pred_neg