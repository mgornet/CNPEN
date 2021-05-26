import numpy as np
from tqdm.notebook import tqdm


# TESTING LOOP
##########################################################################################

def testing(test_loader, device, epochs, model, criterion):
    
    total_loss = []
    total_running_loss = []
    # TO DO remove epoch loop
    for epoch in tqdm(range(epochs), desc="Epochs"):
        running_loss = []
        for step, (anchor_img, positive_img, negative_img) in enumerate(tqdm(test_loader, desc="Processing", leave=False)):
            anchor_img = anchor_img.to(device)
            positive_img = positive_img.to(device)
            negative_img = negative_img.to(device)

            anchor_out = model(anchor_img)
            positive_out = model(positive_img)
            negative_out = model(negative_img)
            
            loss = criterion(anchor_out, positive_out, negative_out)
            
            running_loss.append(loss.cpu().detach().numpy())
            total_running_loss.append(loss.cpu().detach().numpy())
# Remove mean and concatenate instead
        total_loss.append(np.mean(running_loss))

        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, np.mean(running_loss)))
    
    return total_loss, total_running_loss


# TRAINING LOOP
##########################################################################################

def training(model, device, optimizer, criterion, epochs, train_loader, valid_loader):

    total_train_loss = []
    total_valid_loss = []

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
            
            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            
            running_train_loss.append(loss.cpu().detach().numpy())

        running_running_valid_loss = []

        # TO DO virer les running loss

        for _, (anchor_valid, positive_valid, negative_valid) in enumerate(valid_loader):

            anchor_valid = anchor_valid.to(device)
            positive_valid = positive_valid.to(device)
            negative_valid = negative_valid.to(device)

            anchor_valid_out = model(anchor_valid)
            positive_valid_out = model(positive_valid)
            negative_valid_out = model(negative_valid)

            valid_loss = criterion(anchor_valid_out, positive_valid_out, negative_valid_out)
            
            running_running_valid_loss.append(valid_loss.cpu().detach().numpy())
        
        running_valid_loss.append(np.mean(running_running_valid_loss))

        total_valid_loss.append(np.mean(running_valid_loss))

        total_train_loss.append(np.mean(running_train_loss))

        print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, np.mean(running_train_loss)))

    return model, total_train_loss, total_valid_loss