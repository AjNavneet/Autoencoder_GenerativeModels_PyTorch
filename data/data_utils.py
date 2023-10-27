# Importing required libraries
# MNIST data from Torch
from torchvision.datasets import MNIST

# To load the data
from torch.utils.data import DataLoader

# To transform the data
from torchvision import transforms

# Function to transform and load the data with Torch
def get_dl(batch_size):
    # Define the transformation to convert data to Tensors
    train_transforms = transforms.Compose([transforms.ToTensor()])

    # Downloading the training and testing MNIST Data and transforming it to tensors
    train_data = MNIST(root="./train/", train=True, download=True, transform=train_transforms)
    test_data = MNIST(root="./test/", train=False, download=True, transform=train_transforms)
    
    # Accessing the data and label of the first element in the dataset
    data, label = train_data[0]

    # Print the shape and label for the first element of the dataset (commented out)
    # print(data.shape)
    # print(label)

    # Creating DataLoader objects to load the data in batches
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True, drop_last=True)
    
    # Returning the training and testing data loaders
    return train_loader, test_loader
