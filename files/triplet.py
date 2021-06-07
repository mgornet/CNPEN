import random
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
import itertools

# LOSS
#################################################################################

class TripletLoss(nn.Module):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1) 
        distance_negative = (anchor - negative).pow(2).sum(1) 
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class TripletLossRaw(nn.Module):

    def __init__(self, margin):
        super(TripletLossRaw, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1) 
        distance_negative = (anchor - negative).pow(2).sum(1) 
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses

def compute_distances(df, device, model, Xa, Xp, Xn):

    anchors_emb = []
    positives_emb = []
    negatives_emb = []
    
    for i in range(len(Xa)):

    	anchor = torch.reshape(df.Img.iloc[Xa[i]].to(device).float(), (1,3,60,60))
    	positive = torch.reshape(df.Img.iloc[Xp[i]].to(device).float(), (1,3,60,60))
    	negative = torch.reshape(df.Img.iloc[Xn[i]].to(device).float(), (1,3,60,60))

    	anchor_out = model(anchor)
    	positive_out = model(positive)
    	negative_out = model(negative)

    	anchors_emb.append(anchor_out)
    	positives_emb.append(positive_out)
    	negatives_emb.append(negative_out)

    AP = torch.zeros((len(Xa),len(Xp)))
    AN = torch.zeros((len(Xa),len(Xn)))

    for i in range(len(Xa)):

    	anchor_emb = anchors_emb[i]

    	for j in range(len(Xp)):
    		positive_emb = positives_emb[j]
    		distance_positive = (anchor_emb - positive_emb).pow(2).sum(1)
    		AP[i,j] = distance_positive

    	for k in range(len(Xn)):
    		negative_emb = negatives_emb[k]
    		distance_negative = (anchor_emb - negative_emb).pow(2).sum(1)
    		AN[i,k] = distance_negative

    return AP, AN


# TRIPLET GENERATOR
#################################################################################

class TripletGenerator(nn.Module):

    def __init__(self, df, batch_size, device, model, margin, transform=False, mining="standard"):
        
        super(TripletGenerator, self).__init__()
        
        self.batch_size = batch_size
        
        self.df = df

        all_imgs = torch.zeros((len(self.df), 3, 60, 60))
        for i in range(len(self.df)) :
        	all_imgs[i]=self.df.Img.iloc[i]

        self.imgs = all_imgs

        self.classid = df.Classid.unique()
        self.num_samples = len(df.Classid.unique())

        self.device = device
        self.model = model
        self.margin = margin

        self.transform = transform
        self.mining = mining

        Xa, Xp, Xn = [],[],[]

        for classid_unique in df.Classid.unique():
            id_imgs = df.index[df.Classid==classid_unique]
            if len(id_imgs)>1:
                itert = list(itertools.combinations(id_imgs, 2))
                random_index = random.randint(0,len(itert)-1)
                Xa.append(itert[random_index][0])
                Xp.append(itert[random_index][1])
                classids_n = list(df.Classid.unique())
                classids_n.remove(5)
                random_index2 = random.randint(0,len(classids_n)-1)
                classid_n = classids_n[random_index2]
                id_imgs_n = df.index[df.Classid==classid_n]
                random_index3 = random.randint(0,len(id_imgs_n)-1)
                id_img_n = id_imgs_n[random_index3]
                Xn.append(id_img_n)

        self.Xa = Xa
        self.Xp = Xp
        self.Xn = Xn

    def __len__(self):
        return self.num_samples // self.batch_size
        
    def __getitem__(self, batch_index):

        if self.transform :
            image_transforms = transforms.Compose(
                  [
                      transforms.RandomHorizontalFlip(p=0.5)
                  ]
              )

        low_index = batch_index * self.batch_size
        high_index = (batch_index + 1) * self.batch_size

        id_batch_a = self.Xa[low_index:high_index]  # Anchors
        id_batch_p = self.Xp[low_index:high_index]  # Positives
        id_batch_n = self.Xn[low_index:high_index]  # Negatives

        imgs_a = self.imgs[id_batch_a]
        imgs_p = self.imgs[id_batch_p]
        imgs_n = self.imgs[id_batch_n]

        if self.transform :
            imgs_a=image_transforms(imgs_a)
            imgs_p=image_transforms(imgs_p)
            imgs_n=image_transforms(imgs_n)

        return (imgs_a, imgs_p, imgs_n)
    
    
# NETWORK
#################################################################################

class TripletLearner(nn.Module):
    
    def __init__(self):
        super(TripletLearner, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=2),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.fc = nn.Sequential(
            nn.Linear(in_features=15 * 15 * 32, out_features=40),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features= 40, out_features=64)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 32*15*15)
        x = self.fc(x)
        return x