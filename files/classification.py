import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import torch
from triplet import distance, distance_vectors


# CLASSIFICATION & PREDICTIONS
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

# FAIRNESS
###############################################################################

def build_df_fairness(all_imgs, df, gen, epochs, device, model):

    id_a_list, id_p_list, id_n_list = [], [], []
    pos_dist_list, neg_dist_list = [], []

    for _, epoch in enumerate(tqdm(range(epochs),desc="Epoch", leave=False)):
        for _, n_batch in enumerate(tqdm(range(len(gen)),desc="N Batch", leave=False)):

            id_a, id_p, id_n = gen[n_batch]

            id_a_list.extend(id_a)
            id_p_list.extend(id_p)
            id_n_list.extend(id_n)

            anchor_img = all_imgs[id_a].to(device)
            positive_img = all_imgs[id_p].to(device)
            negative_img = all_imgs[id_n].to(device)

            anchor_out = model(anchor_img)
            positive_out = model(positive_img)
            negative_out = model(negative_img)

            pos_dist_list.extend(distance(anchor_out,positive_out).cpu().detach().tolist())     
            neg_dist_list.extend(distance(anchor_out,negative_out).cpu().detach().tolist())

    dist_list = pos_dist_list + neg_dist_list
    A_list = id_a_list + id_a_list
    B_list = id_p_list + id_n_list

    y_pos = [1 for _ in range(len(pos_dist_list))]
    y_neg = [0 for _ in range(len(neg_dist_list))]
    y = y_pos + y_neg

    A_Male, B_Male = [], []
    A_White, B_White = [], [] 

    for id_A, id_B in zip(A_list, B_list):
        A = df.loc[id_A]
        B = df.loc[id_B]
        A_Male.append(A.Male)
        B_Male.append(B.Male)
        A_White.append(A.White)
        B_White.append(B.White)

    df_fairness = pd.DataFrame(list(zip(A_list, B_list, y, dist_list, A_Male, B_Male, A_White, B_White)), columns = ['id_A', 'id_B', 'y_true', 'Distance', 'A_Male', 'B_Male', 'A_White', 'B_White'])

    return df_fairness