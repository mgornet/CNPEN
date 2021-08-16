import numpy as np
from tqdm.notebook import tqdm
import time
import torch
from triplet import distance, distance_vectors


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

def triplet_acc(loader, device, model):

    count_satisfy_condition=0
    total_count=0

    for step, (anchor_img, positive_img, negative_img) in \
    enumerate(tqdm(loader, desc="Processing", leave=False)):

        anchor_img = anchor_img.to(device)
        positive_img = positive_img.to(device)
        negative_img = negative_img.to(device)

        anchor_out = model(anchor_img)
        positive_out = model(positive_img)
        negative_out = model(negative_img)

        for i in range(len(anchor_out)):

            pos_dist = distance_vectors(
                anchor_out[i],positive_out[i]
            ).cpu().detach().item()
            neg_dist = distance_vectors(
                anchor_out[i],negative_out[i]
            ).cpu().detach().item()

            if pos_dist < neg_dist:
                count_satisfy_condition += 1

            total_count += 1

    return count_satisfy_condition/total_count