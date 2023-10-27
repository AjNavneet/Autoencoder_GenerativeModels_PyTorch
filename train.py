# Importing required libraries
import torch
import numpy as np

# Function to train the model
def train_model(no_of_epoch, model, dataloaders, optimizer, lossfn):
    # Setting the device (GPU if available, otherwise CPU)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    best_loss = 100000000.0

    # For each epoch, the model will be trained or evaluated
    for epoch in range(no_of_epoch):
        print('Epoch {}/{}'.format(epoch + 1, no_of_epoch))
        print('-' * 10)
        for phase in ['train', 'valid']:
            train_loss = []
            valid_loss = []
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            for inputs, labels in dataloaders[phase]:
                # Putting inputs and labels on the device (CPU or GPU)
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # Training the model with the given inputs if the phase is 'train'
                    gen_img, dec_model = model(inputs)
                    # Calculating the loss
                    loss = lossfn(gen_img, inputs)
                    if phase == 'train':
                        # Backpropagation during training
                        loss.backward()
                        optimizer.step()
                        train_loss.append(loss.detach().cpu().numpy())
                    else:
                        valid_loss.append(loss.detach().cpu().numpy())
            # Printing the loss
            if phase == 'train':
                print("Train Loss: {}".format(np.mean(train_loss)))
            else:
                print("Valid Loss: {}".format(np.mean(valid_loss)))

    return model, dec_model
