import numpy as np
from tqdm.notebook import tqdm
import wandb
import time
import torch
from torch import optim
from triplet import TripletGenerator, TripletLearner, \
TripletLoss, TripletLossRaw
from torch.utils.data import DataLoader, Dataset
from triplet import distance


# PATH = "./CNPEN/files/"

# TESTING LOOP
###############################################################################

def testing(test_loader, device, model, criterion):
    
    total_loss = []

    for step, (anchor_img, positive_img, negative_img) \
    in enumerate(tqdm(test_loader, desc="Processing", leave=False)):
        anchor_img = anchor_img.to(device)
        positive_img = positive_img.to(device)
        negative_img = negative_img.to(device)

        anchor_out = model(anchor_img)
        positive_out = model(positive_img)
        negative_out = model(negative_img)
        
        loss = criterion(anchor_out, positive_out, negative_out)
        loss = loss.cpu().detach().tolist()
        
        total_loss += loss
    
    return total_loss

def compute_distances(loader, device, model):
    
    list_distance_pos = []
    list_distance_neg = []

    for step, (anchor_img, positive_img, negative_img) \
    in enumerate(tqdm(loader, desc="Processing", leave=False)):
        anchor_img = anchor_img.to(device)
        positive_img = positive_img.to(device)
        negative_img = negative_img.to(device)

        anchor_out = model(anchor_img)
        positive_out = model(positive_img)
        negative_out = model(negative_img)
        
        dist_pos = distance(anchor_out, positive_out).cpu().detach().tolist()
        dist_neg = distance(anchor_out, negative_out).cpu().detach().tolist()
        
        list_distance_pos += dist_pos
        list_distance_neg += dist_neg
    
    return list_distance_pos, list_distance_neg


# TRAINING LOOP
###############################################################################

def training(model, device, optimizer, criterion, epochs, 
    train_loader, valid_loader, save_epoch=True):

    total_train_loss = []
    total_valid_loss = []

    nb_step_train = len(train_loader)
    nb_step_valid = len(valid_loader)

    model.train()

    for epoch in tqdm(range(epochs), desc="Epochs"):

        running_train_loss = []
        running_valid_loss = []

        for step, (anchor_img, positive_img, negative_img) \
        in enumerate(tqdm(train_loader, desc="Training", leave=False)):

            anchor_img = anchor_img.to(device)
            positive_img = positive_img.to(device)
            negative_img = negative_img.to(device)

            optimizer.zero_grad()

            anchor_out = model(anchor_img)
            positive_out = model(positive_img)
            negative_out = model(negative_img)

            train_loss = criterion(anchor_out, positive_out, negative_out)
            train_loss.backward()
            optimizer.step()
      
            train_step = step + nb_step_train * epoch
            
            wandb.log({"training step":train_step, "training loss":train_loss})

            running_train_loss.append(train_loss.cpu().detach().numpy())

        for step, (anchor_valid, positive_valid, negative_valid) \
        in enumerate(tqdm(valid_loader, desc="Evaluating", leave=False)):

            anchor_valid = anchor_valid.to(device)
            positive_valid = positive_valid.to(device)
            negative_valid = negative_valid.to(device)

            anchor_valid_out = model(anchor_valid)
            positive_valid_out = model(positive_valid)
            negative_valid_out = model(negative_valid)

            valid_loss = criterion(
                anchor_valid_out, positive_valid_out, negative_valid_out
            )

            valid_step = step + nb_step_valid * epoch
            
            wandb.log({
                "validation step":valid_step, "validation loss":valid_loss
            })

            running_valid_loss.append(valid_loss.cpu().detach().numpy())

        mean_train_loss = np.mean(running_train_loss)
        mean_valid_loss = np.mean(running_valid_loss)

        total_train_loss.append(mean_train_loss)
        total_valid_loss.append(mean_valid_loss)

        wandb.log({
            "epoch":epoch,
            "mean training loss":mean_train_loss,
            "mean validation loss":mean_valid_loss
        })
        print("Epochs: {}/{} - Loss: {:.4f}".format(
            epoch+1, epochs, mean_train_loss))

        # save training checkpoint
        if save_epoch :
            dt = time.strftime("%Y_%m_%d-%H_%M_%S")
            fn = "./Log" + str(dt) + str("-") + str(epoch) + "_checkpoint.pt"

            info_dict = { 'epoch' : epoch,
                'torch_random_state' : torch.random.get_rng_state(),
                'numpy_random_state' : np.random.get_state(),
                'model_state' : model.state_dict(),
                'optimizer_state' : optimizer.state_dict(),
                'mean_training_loss':mean_train_loss,
                'mean_validation_loss':mean_valid_loss }

            torch.save(info_dict, fn)

    return model


def adaptative_train(model, device, optimizer, criterion, epochs, df_train,
    df_valid, BATCH_SIZE, BATCH_VALID_SIZE, margin, all_imgs, save_phase=True):

    phase = 1

    gen_train = TripletGenerator(
        df_train, all_imgs, BATCH_SIZE, device, model, margin, transform = True
    )
    train_loader = DataLoader(gen_train, batch_size=None, shuffle=True)

    gen_valid = TripletGenerator(
        df_valid, all_imgs, BATCH_VALID_SIZE, device, model, margin
    )
    valid_loader = DataLoader(gen_valid, batch_size=None, shuffle=True)

    model = training(
        model, device, optimizer, criterion, epochs,
        train_loader, valid_loader, save_epoch=False
    )

    if save_phase:
        dt = time.strftime("%Y_%m_%d-%H_%M_%S")
        fn = "./Log" + str(dt) + str("-") + str(phase) + "_checkpoint.pt"

        info_dict = { 'phase' : phase,
            'torch_random_state' : torch.random.get_rng_state(),
            'model_state' : model.state_dict(),
            'optimizer_state' : optimizer.state_dict() }

        torch.save(info_dict, fn)

    phase = 2

    gen_train = TripletGenerator(
        df_train, all_imgs, BATCH_SIZE, device,
        model, margin, transform = True, mining="semi"
    )
    train_loader = DataLoader(gen_train, batch_size=None, shuffle=True)

    model = training(
        model, device, optimizer, criterion, epochs,
        train_loader, valid_loader, save_epoch=False
    )

    if save_phase:
        dt = time.strftime("%Y_%m_%d-%H_%M_%S")
        fn = "./Log" + str(dt) + str("-") + str(phase) + "_checkpoint.pt"

        info_dict = { 'phase' : phase,
            'torch_random_state' : torch.random.get_rng_state(),
            'model_state' : model.state_dict(),
            'optimizer_state' : optimizer.state_dict() }

        torch.save(info_dict, fn)

    phase = 3

    gen_train = TripletGenerator(
        df_train, all_imgs, BATCH_SIZE, device, model,
        margin, transform = True, mining="hard"
    )
    train_loader = DataLoader(gen_train, batch_size=None, shuffle=True)

    model = training(
        model, device, optimizer, criterion, epochs,
        train_loader, valid_loader, save_epoch=False
    )

    if save_phase:
        dt = time.strftime("%Y_%m_%d-%H_%M_%S")
        fn = "./Log" + str(dt) + str("-") + str(phase) + "_checkpoint.pt"

        info_dict = { 'phase' : phase,
            'torch_random_state' : torch.random.get_rng_state(),
            'model_state' : model.state_dict(),
            'optimizer_state' : optimizer.state_dict() }

        torch.save(info_dict, fn)

    return model

# def tuning_lr(model, device, criterion, epochs, 
#     train_loader, valid_loader):

#     lr = 1e-3
#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     model = training(
#         model, device, optimizer, criterion, epochs,
#         train_loader, valid_loader, save_epoch=False
#     )

def adaptative_train_lr(model, device, criterion,
    train_loader, valid_loader):

    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epochs = 300

    model = training(
        model, device, optimizer, criterion, epochs,
        train_loader, valid_loader, save_epoch=False
    )

    epochs = 100
    for _ in range(7):

        lr = lr/2
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model = training(
            model, device, optimizer, criterion, epochs,
            train_loader, valid_loader, save_epoch=False
        )

    return model