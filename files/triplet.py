import random
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms

# LOSS
#################################################################################

# TO DO Loss for testing without mean
class TripletLoss(nn.Module):

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1) 
        distance_negative = (anchor - negative).pow(2).sum(1) 
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


# TRIPLET GENERATOR
#################################################################################

class TripletGenerator(nn.Module):
    def __init__(self, Xa_train, Xp_train, batch_size, all_imgs, neg_imgs_idx):
        self.cur_img_index = 0
        self.cur_img_pos_index = 0
        self.batch_size = batch_size
        
        self.imgs = all_imgs
        self.Xa = Xa_train  # Anchors
        self.Xp = Xp_train  # Positives
        self.cur_train_index = 0
        self.num_samples = Xa_train.shape[0]
        self.neg_imgs_idx = neg_imgs_idx
        
    def __len__(self):
        return self.num_samples // self.batch_size
        
    def __getitem__(self, batch_index):

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