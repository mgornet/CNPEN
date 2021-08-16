import random
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms
import itertools
import warnings
# torch.multiprocessing.set_start_method('spawn') # to prevent error when using semi or hard mining --> empêche l'entrainement de se lancer, patine

# LOSS
###############################################################################

class TripletLoss(nn.Module):

    """
    A class to represent the triplet loss function.
    L(A,P,N) = max( ||f(A)-f(P)||² - ||f(A)-f(N)||² + m, 0)

    Attributes
    ----------
    margin: float
        m in the function above

    Methods
    -------
    forward(anchor, positive, negative, size_average=True):
        Compute the loss in form of a float.
        If size_average=True, take the mean, else, returns the sum.
    """

    def __init__(self, margin):
        """
        Constructs all the necessary attributes for the TripletLoss object.

        Parameters
        ----------
            margin: float
                m in the function above
        """
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):

        """Inputs:
               anchor: tensor of the anchor image
               positive: tensor of the positive image
               negative: tensor of the negative image
               size_average: bool.
               If true, take the mean of the loss, if false, returns the sum

           Output:
               float (either mean of sum of the losses)"""

        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

def distance(a,b):
    """Compute the euclidian distance between embeddings"""
    return (a - b).pow(2).sum(1)

def distance_vectors(a,b):
    return (a - b).pow(2).sum()

class TripletLossRaw(nn.Module):

    """
    A class to represent the triplet loss function.
    L(A,P,N) = max( ||f(A)-f(P)||² - ||f(A)-f(N)||² + m, 0)

    Attributes
    ----------
    margin: float
        m in the function above

    Methods
    -------
    forward(anchor, positive, negative):
        Compute the loss in form of a vector
    """

    def __init__(self, margin):
        super(TripletLossRaw, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """Inputs:
               anchor: tensor of the anchor image
               positive: tensor of the positive image
               negative: tensor of the negative image

           Output:
               torch tensor"""
        distance_positive = (anchor - positive).pow(2).sum(1) 
        distance_negative = (anchor - negative).pow(2).sum(1) 
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses

def compute_distances(all_imgs, device, model, Xa, Xp, Xn):

    """Inputs:
            all_imgs: torch tensor of shape (nb_imgs,3,60,60)
            with values in (0,1) containing all dataset images
            device: cuda or gpu
            model: TripletLearner network
            Xa: tensor of the anchor images
            Xp: tensor of the positive images
            Xn: tensor of the negative images

        Output:
            AP: tensor of the distances between
            all anchor images and all positive images in the batch
            AN: tensor of the distances between
            all anchor images and all negative images in the batch"""

    anchors_emb = []
    positives_emb = []
    negatives_emb = []
    
    for i in range(len(Xa)):

        anchor = torch.reshape(all_imgs[Xa[i]].to(device), (1,3,60,60))
        positive = torch.reshape(all_imgs[Xp[i]].to(device), (1,3,60,60))
        negative = torch.reshape(all_imgs[Xn[i]].to(device), (1,3,60,60))

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
###############################################################################

AUGMENT = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation((-5,5)),
    transforms.RandomResizedCrop(size=60, scale=(0.7,1.), ratio=(1.,1.)),
    transforms.Compose([
        transforms.Normalize(mean=[0,0,0],std=[255,255,255]),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
        transforms.Normalize(mean=[0,0,0],std=[1/255,1/255,1/255])
    ])
])

class TripletGenerator(nn.Module):

    """
    A class to represent the triplet generator.

    Attributes
    ----------
    batch_size: int. Size of the batch
    df: pandas dataframe
    imgs: tensor of images
    num_samples: number of classid with more than one image
    device: cpu or gpu
    model: TripletLearner model
    margin: float 
    tranform: bool. If true, apply augmentation
    mining: str. "standard", "semi" or "hard"
    apply augmentation: transform function

    Methods
    -------
    """

    def __init__(
        self, df, all_imgs, batch_size, device, model, margin,
        transform=False, apply_augmentation=AUGMENT, mining="standard"):
        
        super(TripletGenerator, self).__init__()
        
        self.df = df

        self.imgs = all_imgs

        value_count = df.Classid.value_counts()
        self.id_list = list(value_count[value_count.values>1].index)

        self.num_samples = len(self.id_list)

        if self.num_samples<batch_size :
            warnings.warn(
                f'Batch size number was changed from {batch_size} to '
                f'{self.num_samples} because there are only {self.num_samples}'
                f' individuals with more than 2 pictures.'
            )
            batch_size=self.num_samples

        self.batch_size = batch_size

        if self.num_samples % batch_size != 0:
            warnings.warn(
                f'Number of unique identities with more than 2 pictures '
                f'({self.num_samples}) is not divisible by batch_size '
                f'({batch_size}). Remainder: {self.num_samples % batch_size}'
            )

        self.device = device
        self.model = model
        self.margin = margin

        self.transform = transform
        self.mining = mining

        self.apply_augmentation = apply_augmentation

        random.shuffle(self.id_list)
        self.last_batch_index = len(self)-1

    def __len__(self):
        return self.num_samples // self.batch_size
        
    def __getitem__(self, batch_index):

        low_index = batch_index * self.batch_size
        high_index = (batch_index + 1) * self.batch_size

        classid_batch = self.id_list[low_index:high_index]

        Xa = []
        Xp = []
        Xn = []

        for classid_unique in classid_batch:
            id_imgs = list(self.df.index[self.df.Classid==classid_unique])
            # choose one img at random inside this classid
            # rq: if k=1 returns a list of one elmt, hence the [0]
            xa = random.choices(id_imgs)[0]
            Xa.append(xa)
            # remove it from the list of possible imgs
            id_imgs.remove(xa)
            # take another image
            # rq: if we take xa, xp = random.choices(id_imgs, k=2),
            # we end up with the same imgs sometimes
            xp = random.choices(id_imgs)[0]
            Xp.append(xp)
            # random img among all
            xn = random.choices(self.df.index)[0]
            # TO DO check xn classid diff from xa classid if not try again
            while self.df.loc[xn].Classid == self.df.loc[xa].Classid:
                xn = random.choices(self.df.index)[0]
            Xn.append(xn)

        if self.mining == "semi":
            AP, AN = compute_distances(
                self.imgs, self.device, self.model, Xa, Xp, Xn
                )
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
            AP, AN = compute_distances(
                self.imgs, self.device, self.model, Xa, Xp, Xn
                )
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

        if batch_index == self.last_batch_index:
            random.shuffle(self.id_list)

        return (imgs_a, imgs_p, imgs_n)

# NETWORK
###############################################################################

class TripletLearner(nn.Module):

    """
    A class to represent the image encoder.

    Attributes
    ----------
    base_channels: int
    conv: nn.Sequential
    avg: nn.AvgPool2d
    fc: nn.Sequential

    Methods
    -------
    forward
    """

    def __init__(self, base_channels=32, dropout=0):
        self.base_channels = base_channels
        self.dropout = dropout

        super(TripletLearner, self).__init__()
        self.conv = nn.Sequential(
            # in the case where base_channels=16
            # (3,60,60)
            nn.Conv2d(
                in_channels=3, out_channels=base_channels,
            	kernel_size=3, padding=1),
            nn.ReLU(),
            # (16,60,60)
            nn.Conv2d(
                in_channels=base_channels, out_channels=base_channels,
                kernel_size=3, padding=1),
            nn.ReLU(),
            # (16,60,60)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (16,30,30)
            nn.Conv2d(
                in_channels=base_channels, out_channels=base_channels*2,
                kernel_size=3, padding=1),
            nn.ReLU(),
            # (32,30,30)
            nn.Conv2d(
                in_channels=base_channels*2, out_channels=base_channels*2,
                kernel_size=3, padding=1),
            nn.ReLU(),
            # (32,30,30)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (32,15,15)
            nn.Conv2d(
                in_channels=base_channels*2, out_channels=base_channels*4,
                kernel_size=3, padding=1),
            nn.ReLU(),
            # (64,15,15)
            nn.Conv2d(
                in_channels=base_channels*4, out_channels=base_channels*4,
                kernel_size=3, padding=1),
            nn.ReLU(),
            # (64,15,15)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (64,7,7)
            nn.Conv2d(
                in_channels=base_channels*4, out_channels=base_channels*8,
                kernel_size=3, padding=1),
            nn.ReLU(),
            # (128,7,7)
            nn.Conv2d(
                in_channels=base_channels*8, out_channels=base_channels*8,
                kernel_size=3, padding=1),
            nn.ReLU(),
            # (128,7,7)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # (128,3,3)
            nn.Conv2d(
                in_channels=base_channels*8, out_channels=base_channels*16,
                kernel_size=3, padding=1),
            nn.ReLU(),
            # (256,3,3)
            nn.Conv2d(
                in_channels=base_channels*16, out_channels=base_channels*16,
                kernel_size=3, padding=1),
            nn.ReLU(),
            # (256,3,3)
        )

        # (256,3,3)
        self.avg = nn.AvgPool2d(kernel_size=3)
        # (256,1,1)
        
        # (1,256)
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features= base_channels*16,
                out_features=base_channels*16)
            # (1,256)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = self.avg(x)
        x = x.view(-1, self.base_channels*16)
        x = self.fc(x)
        return x