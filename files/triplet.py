import random
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms

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

    image_transforms = transforms.Compose(
              [
                  transforms.ToTensor(),
              ]
          )
    
    for i in range(len(Xa)):

    	anchor = torch.reshape(image_transforms(df.Img.iloc[Xa[i]]).to(device).float(), (1,3,60,60))
    	positive = torch.reshape(image_transforms(df.Img.iloc[Xp[i]]).to(device).float(), (1,3,60,60))
    	negative = torch.reshape(image_transforms(df.Img.iloc[Xn[i]]).to(device).float(), (1,3,60,60))

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
    def __init__(self, Xa_train, Xp_train, batch_size, df, neg_imgs_idx, device, model, margin, transform=False, mining="standard"):
        
        super(TripletGenerator, self).__init__()

        self.cur_img_index = 0
        self.cur_img_pos_index = 0
        self.batch_size = batch_size
        
        self.df = df
        self.imgs = df.Img.values
        self.Xa = Xa_train  # Anchors
        self.Xp = Xp_train  # Positives
        self.cur_train_index = 0
        self.num_samples = Xa_train.shape[0]
        self.neg_imgs_idx = neg_imgs_idx

        self.device = device
        self.model = model
        self.margin = margin

        self.transform = transform
        self.mining = mining
        
    def __len__(self):
        return self.num_samples // self.batch_size
        
    def __getitem__(self, batch_index):

    	if self.transform :
            image_transforms = transforms.Compose(
                  [
                      transforms.ToTensor(),
                      transforms.RandomHorizontalFlip(p=0.5)
                  ]
              )

    	else :
            image_transforms = transforms.Compose(
                  [
                      transforms.ToTensor(),
                  ]
              )

    	low_index = batch_index * self.batch_size
    	high_index = (batch_index + 1) * self.batch_size

    	imgs_a = self.Xa[low_index:high_index]  # Anchors
    	imgs_p = self.Xp[low_index:high_index]  # Positives
    	imgs_n = random.sample(self.neg_imgs_idx, imgs_a.shape[0])  # Negatives

    	if mining == "semi":
    		AP, AN = compute_distances(self.df, self.device, self.model, imgs_a, imgs_p, imgs_n)
    		imgs_n_ordered = list(imgs_n)
    		for i in range(len(imgs_a)):
    			satisfy_cond = []
    			for k in range(len(imgs_n)):
    				if (AP[i,i]<AN[i,k])&(AN[i,k]<self.margin):
    					satisfy_cond.append(imgs_n[k])
    			imgs_n_ordered[i]=imgs_n[argmin(satisfy_cond[i])]
    		imgs_n = imgs_n_ordered

    	elif mining == "hard":
    		AP, AN = compute_distances(self.df, self.device, self.model, imgs_a, imgs_p, imgs_n)
    		imgs_n_ordered = list(imgs_n)
    		for i in range(len(imgs_a)):
    			imgs_n_ordered[i]=imgs_n[argmin(AN[i])]  
    		imgs_n = imgs_n_ordered

    	imgs_a = self.imgs[imgs_a]
    	imgs_p = self.imgs[imgs_p]
    	imgs_n = self.imgs[imgs_n]

    	anchors = torch.zeros((self.batch_size,3,60,60))
    	positives = torch.zeros((self.batch_size,3,60,60))
    	negatives = torch.zeros((self.batch_size,3,60,60))

    	for batch in range(self.batch_size):
    		anchors[batch]=image_transforms(imgs_a[batch])
    		positives[batch]=image_transforms(imgs_p[batch])
    		negatives[batch]=image_transforms(imgs_n[batch])
            
    	return (anchors, positives, negatives)
    
    
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