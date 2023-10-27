# Import required libraries
import torch.nn as nn
import torch
from torchinfo import summary

# Encoder for Autoencoder
class Encoder(nn.Module):
    def __init__(self, batch_size):
        super().__init__()
        self.batchsize = batch_size
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, 2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, padding=1)
        self.LRelu = nn.LeakyReLU()
        self.fc1 = nn.Linear(3136, 2)

    # Forward propagation function
    def forward(self, x):
        layer1 = self.LRelu(self.conv1(x))
        layer2 = self.LRelu(self.conv2(layer1))
        layer3 = self.LRelu(self.conv3(layer2))
        layer4 = self.LRelu(self.conv4(layer3))
        flat = layer4.view(self.batchsize, -1)
        flat_shape = flat.size()[1]
        encoder_out = self.fc1(flat)
        return encoder_out

# Decoder for Autoencoder
class Decoder(nn.Module):
    def __init__(self, batch_size):
        super().__init()
        self.batch_size = batch_size
        self.fc1 = nn.Linear(2, 3136)
        self.Dconv1 = nn.ConvTranspose2d(64, 64, 3, 1, padding=1)
        self.Dconv2 = nn.ConvTranspose2d(64, 64, 3, 2, padding=1, output_padding=1)
        self.Dconv3 = nn.ConvTranspose2d(64, 32, 3, 2, padding=1, output_padding=1)
        self.Dconv4 = nn.ConvTranspose2d(32, 1, 3, 1, padding=1)
        self.LRelu = nn.LeakyReLU()

    # Forward propagation function
    def forward(self, x):
        fc = self.fc1(x)
        reshaped = fc.view(self.batch_size, 64, 7, 7)
        layer1 = self.LRelu(self.Dconv1(reshaped))
        layer2 = self.LRelu(self.Dconv2(layer1))
        layer3 = self.LRelu(self.Dconv3(layer2))
        layer4 = self.Dconv4(layer3)
        out = torch.sigmoid(layer4)
        return out

class Encoder_Decoder_model(nn.Module):
    def __init__(self, batch_size):
        super().__init()
        self.batch_size = batch_size

        # Initialize an object for the Encoder
        self.enc = Encoder(self.batch_size)

        # Initialize an object for the Decoder
        self.dec = Decoder(self.batch_size)
        self.sigmoid_act = torch.nn.Sigmoid()

    # Forward propagation function
    def forward(self, img):
        # Input the image into the encoder
        enc_out = self.enc(img)
        
        # Use the encoder output as input for the decoder
        dec_out = self.dec(enc_out)
        out = self.sigmoid_act(dec_out)
        return out, self.dec

# Uncomment the following lines to see model summaries
# Encoder_model = Encoder(10)
# print(summary(Encoder_model, input_size=(10, 1, 28, 28)))

# Decode_model = Decoder(10)
# print(summary(Decode_model, input_size=(10, 2)))
