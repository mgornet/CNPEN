import numpy as np
from tqdm.notebook import tqdm
import wandb


# TESTING LOOP
##########################################################################################

def testing(test_loader, device, model, criterion):
    
    total_loss = []

    for step, (anchor_img, positive_img, negative_img) in enumerate(tqdm(test_loader, desc="Processing", leave=False)):
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


# TRAINING LOOP
##########################################################################################

def training(model, device, optimizer, criterion, epochs, train_loader, valid_loader):

    total_train_loss = []
    total_valid_loss = []

    nb_step_train = len(train_loader)
    nb_step_valid = len(valid_loader)

    model.train()

    for epoch in tqdm(range(epochs), desc="Epochs"):

    	running_train_loss = []
    	running_valid_loss = []

    	for step, (anchor_img, positive_img, negative_img) in enumerate(tqdm(train_loader, desc="Training", leave=False)):

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

    	for step, (anchor_valid, positive_valid, negative_valid) in enumerate(tqdm(valid_loader, desc="Evaluating", leave=False)):

            anchor_valid = anchor_valid.to(device)
            positive_valid = positive_valid.to(device)
            negative_valid = negative_valid.to(device)

            anchor_valid_out = model(anchor_valid)
            positive_valid_out = model(positive_valid)
            negative_valid_out = model(negative_valid)

            valid_loss = criterion(anchor_valid_out, positive_valid_out, negative_valid_out)

            valid_step = step + nb_step_valid * epoch
            
            wandb.log({"validation step":valid_step, "validation loss":valid_loss})

            running_valid_loss.append(valid_loss.cpu().detach().numpy())
            
    	mean_train_loss = np.mean(running_train_loss)
    	mean_valid_loss = np.mean(running_valid_loss)

    	total_train_loss.append(mean_train_loss)
    	total_valid_loss.append(mean_valid_loss)

    	wandb.log({"epoch":epoch, "mean training loss":mean_train_loss, "mean validation loss":mean_valid_loss})
    	print("Epochs: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, mean_train_loss))

    return model