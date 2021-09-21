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

def build_df_fairness(all_imgs, df, gen, epochs, device, model, threshold):

    id_a_list, id_p_list, id_n_list = [], [], []
    pos_dist_list, neg_dist_list = [], []

    for _, epoch in enumerate(tqdm(range(epochs),desc="Epoch", leave=False)):
        for n_batch in range(len(gen)):

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

            pos_dist_list.extend(
                distance(anchor_out,positive_out).cpu().detach().tolist()
            )     
            neg_dist_list.extend(
                distance(anchor_out,negative_out).cpu().detach().tolist()
            )

    dist_list = pos_dist_list + neg_dist_list

    A_list = id_a_list + id_a_list
    B_list = id_p_list + id_n_list

    y_pos = [1 for _ in range(len(pos_dist_list))]
    y_neg = [0 for _ in range(len(neg_dist_list))]
    y_true = y_pos + y_neg

    y_pred = [int(dist_list[i]<threshold) for i in range(len(dist_list))]

    A_Male, B_Male = [], []
    A_White, B_White = [], [] 

    for id_A, id_B in zip(A_list, B_list):
        A = df.loc[id_A]
        B = df.loc[id_B]
        A_Male.append(A.Male)
        B_Male.append(B.Male)
        A_White.append(A.White)
        B_White.append(B.White)
        A_Black.append(A.Black)
        B_Black.append(B.Black)
        A_Asian.append(A.Asian)
        B_Asian.append(B.Asian)
        A_Indian.append(A.Indian)
        B_Indian.append(B.Indian)
        A_Youth.append(A.Youth)
        B_Youth.append(B.Youth)
        A_Senior.append(A.Senior)
        B_Senior.append(B.Senior)
        A_Sunglasses.append(A.Sunglasses)
        B_Sunglasses.append(B.Sunglasses)

    df_fairness = pd.DataFrame(
        list(zip(A_list, B_list, y_true, y_pred, dist_list, A_Male, B_Male,\
            A_White, B_White, A_Black, B_Black, A_Asian, B_Asian, A_Indian, \
            B_Indian, A_Youth, B_Youth, A_Senior, B_Senior, A_Sunglasses,\
            B_Sunglasses)), \
            columns = ['id_A', 'id_B', 'y_true', 'y_pred', 'Distance', \
            'A_Male', 'B_Male', 'A_White', 'B_White', 'A_Black', 'B_Black',\
            'A_Asian', 'B_Asian', 'A_Indian', 'B_Indian', 'A_Youth', 'B_Youth',\
            'A_Senior', 'B_Senior', 'A_Sunglasses', 'B_Sunglasses'])

    # add somme dummy variables
    df_fairness['A_WhiteMale'] = df_fairness['A_White'] * \
        df_fairness['A_Male']
    df_fairness['B_WhiteMale'] = df_fairness['B_White'] * \
        df_fairness['B_Male']
    df_fairness['AB_WhiteMale'] = df_fairness['A_WhiteMale'] * \
        df_fairness['B_WhiteMale']
    df_fairness['AB_NoWhiteMale'] = ((df_fairness['A_WhiteMale']==0) \
        & (df_fairness['B_WhiteMale']==0)).astype(float)
    df_fairness['correct_predict'] = \
        (df_fairness['y_true'] == df_fairness['y_pred']).astype(float)

    return df_fairness

def bootstrap(df, agg_func, num_bootstraps=1000, percentiles=[5,25,50,75,95]):

    results = []
    rng = np.random.default_rng()
    for i in range(num_bootstraps):
        indices = rng.integers(0,len(df),len(df))
        resampled_df = df.iloc[indices]
        results.append(agg_func(resampled_df))
    return np.percentile(results, percentiles)

def triplet_acc_fairness(df_fairness):

    count_satisfy_condition=0
    total_count=0

    demi_len_df = len(df_fairness)//2

    for i in range(demi_len_df):
        dist_pos = df_fairness.iloc[i].Distance
        dist_neg = df_fairness.iloc[i+demi_len_df].Distance
        if dist_pos < dist_neg :
            count_satisfy_condition+=1
        total_count+=1

    return count_satisfy_condition/total_count