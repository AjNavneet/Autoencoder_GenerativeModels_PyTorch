# Importing required libraries
import torch
from model.autoencoder_decoder_model import Encoder_Decoder_model
from data.data_utils import get_dl
from train import train_model

# Setting a seed for reproducibility
torch.manual_seed(0)

# Parameters for model building
batchsize = 32
n_epoch = 10

# Downloading and transforming the data using the get_dl function
train_loader, test_loader = get_dl(batchsize)

# Putting data loaders into a dictionary for training
dl = {}
dl["train"] = train_loader
dl["valid"] = test_loader

# Creating an autoencoder model
model = Encoder_Decoder_model(batchsize)

# Defining the device for model training (using GPU if available, otherwise CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Learning rate
lr = 0.001

# Selecting the optimizer (Adam)
optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-05)

# Mean Squared Error (MSE) loss function
loss_fn = torch.nn.MSELoss()

# Training the model and obtaining the encoder and decoder models
model, dec_model = train_model(n_epoch, model, dl, optim, loss_fn)

# Saving the model's state dictionaries to files
torch.save(model.state_dict(), "encoder_model.pth")
torch.save(dec_model.state_dict(), "decoder_model.pth")
