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

def compute_distances(all_imgs, device, model, Xa, Xp, Xn):

    anchors_emb = []
    positives_emb = []
    negatives_emb = []
    
    for i in range(len(Xa)):

    	anchor = torch.reshape(all_imgs[Xa[i]].to(device), (1,3,60,60))
    	positive = torch.reshape(all_imgs[Xp[i]].to(device), (1,3,60,60))
    	negative = torch.reshape(all_imgs[Xn[i]].to(device), (1,3,60,60))  #.float(), (1,3,60,60))

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

    def __init__(self, df, all_imgs, batch_size, device, model, margin, transform=False, mining="standard", negative=False):
        
        super(TripletGenerator, self).__init__()
        
        self.batch_size = batch_size
        
        self.df = df

        self.imgs = all_imgs

        self.num_samples = len(df.Classid.value_counts()[df.Classid.value_counts().values>1])

        self.device = device
        self.model = model
        self.margin = margin

        self.transform = transform
        self.mining = mining
        self.negative = negative

        self.apply_augmentation = transforms.Compose(
              [
                  transforms.RandomHorizontalFlip(p=0.5),
                  # transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(),]), p=0.3),
                  # transforms.RandomPerspective(),
                  # transforms.RandomCrop(),
                  transforms.RandomRotation((-10,10)),
                  # transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(kernel_size=3),]),p=0.2),
                  # transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2)
              ]
          )

        self.id_list = list(df.Classid.value_counts()[df.Classid.value_counts().values>1].index)
        random.shuffle(self.id_list)

    def __len__(self):
        return self.num_samples // self.batch_size
        
    def __getitem__(self, batch_index):

        low_index = batch_index * self.batch_size
        high_index = (batch_index + 1) * self.batch_size

        classid_batch = self.id_list[low_index:high_index]

        Xa = []
        Xp = []
        Xn = []

        if self.negative == True:
            for classid_unique in classid_batch:
                random_ids = [random.randint(self.df.index.min(),self.df.index.max()) for _ in range(3)]
                Xa.append(random_ids[0])
                Xp.append(random_ids[1])
                Xn.append(random_ids[2])

        elif self.negative == False:
            for classid_unique in classid_batch:
                id_imgs = self.df.index[self.df.Classid==classid_unique]
                itert = list(itertools.combinations(id_imgs, 2))
                random_index = random.randint(0,len(itert)-1)
                Xa.append(itert[random_index][0])
                Xp.append(itert[random_index][1])
                # list of all classids without the one already used by anchor
                classids_n = list(self.df.Classid.unique())
                classids_n.remove(classid_unique)
                # randomly select one identity for the negative image
                random_index2 = random.randint(0,len(classids_n)-1)
                classid_n = classids_n[random_index2]
                # select a random image amonst this identity
                id_imgs_n = self.df.index[self.df.Classid==classid_n]
                random_index3 = random.randint(0,len(id_imgs_n)-1)
                id_img_n = id_imgs_n[random_index3]
                Xn.append(id_img_n)

        if self.mining == "semi":
            AP, AN = compute_distances(self.imgs, self.device, self.model, Xa, Xp, Xn)
            Xn_ordered = list(Xn)
            for i in range(len(Xa)):
                satisfy_cond = []
                for k in range(len(Xn)):
                    if (AP[i,i]<AN[i,k])&(AN[i,k]<self.margin):
                        satisfy_cond.append(Xn[k])
                if len(satisfy_cond)>0 :
                    Xn_ordered[i]=Xn[np.argmin(satisfy_cond)]
                else :
                    Xn_ordered[i]=Xn[i]
                Xn = Xn_ordered

        elif self.mining == "hard":
            AP, AN = compute_distances(self.imgs, self.device, self.model, Xa, Xp, Xn)
            Xn_ordered = list(Xn)
            for i in range(len(Xa)):
                Xn_ordered[i]=Xn[torch.argmin(AN[i])]  
            Xn = Xn_ordered

        imgs_a = self.imgs[Xa]
        imgs_p = self.imgs[Xp]
        imgs_n = self.imgs[Xn]

        if self.transform :
            imgs_a=self.apply_augmentation(imgs_a)
            imgs_p=self.apply_augmentation(imgs_p)
            imgs_n=self.apply_augmentation(imgs_n)

        return (imgs_a, imgs_p, imgs_n)
    
    
# NETWORK
#################################################################################

class TripletLearner(nn.Module):
    
    def __init__(self):
        super(TripletLearner, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features= 256, out_features=512)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x