import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights



class TransposeBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransposeBasicBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout(0.2)
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=2, output_padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        # Perform upsampling path
        upsampled = self.upsample(x)

        return self.relu(out + upsampled)

class ResNetEncDecSubNet(nn.Module):
    def __init__(self):
        super(ResNetEncDecSubNet, self).__init__()

        # Encoder: Using ResNet layers up to layer4
        # resnet = models.resnet18(pretrained=True)
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Encoder layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Reduce channels from 512 to 256
        self.reduce_channels = nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False)

        # Note: decrease the complexity on the model
        # Note: train more epochs
        
        # NIN layer (256 -> 128 -> 64)
        self.nin = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(128, 64, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )

        # Decoder (TransposeBasicBlock layers)
        self.uplayer1 = TransposeBasicBlock(64, 64)
        self.uplayer2 = TransposeBasicBlock(64, 32)
        self.uplayer3 = TransposeBasicBlock(32, 16)

        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2.0, mode='bilinear')

        # Final conv layer
        self.convtranspose1 = nn.ConvTranspose2d(16, 3, kernel_size=7, stride=2, padding=3, output_padding=1, bias=False)

        # Final activation (Identity in this case)
        self.final_activation = nn.Identity()

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Reduce channels
        x = self.reduce_channels(x)

        # NIN Layer
        x = self.nin(x)

        # Decoder layers
        decoded = self.uplayer1(x)
        decoded = self.uplayer2(decoded)
        decoded = self.uplayer3(decoded)

        # Upsample
        decoded = self.upsample(decoded)

        # Final convolution
        decoded = self.convtranspose1(decoded)

        # Final activation (here it's Identity)
        decoded = self.final_activation(decoded)

        # print("decoded.shape, ", decoded.shape)
        return decoded
    
class DiscriminativeSubNet(nn.Module):
    def __init__(self):
        super(DiscriminativeSubNet, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1),  # Input: Original + Reconstructed (3+3=6 channels)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),  # Output: 1 channel anomaly map
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

    def forward(self, original, reconstructed):
        # Concatenate along channel dimension
        x = torch.cat((original, reconstructed), dim=1) 
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

class ResNetEncDecWithDiscrimination(nn.Module):
    def __init__(self):
        super(ResNetEncDecWithDiscrimination, self).__init__()
        
        # reconstruction network
        self.reconstruction_net = ResNetEncDecSubNet()
        
        # discriminative network
        self.discriminative_net = DiscriminativeSubNet()
    
    def forward(self, x):
        # Reconstruction
        reconstructed = self.reconstruction_net(x)
        
        # print(f"Original shape: {x.shape}, Reconstructed shape: {reconstructed.shape}")

        # Anomaly Map
        anomaly_map = self.discriminative_net(x, reconstructed)
        

        return reconstructed, anomaly_map
    

if __name__ == "__main__":
    model = ResNetEncDecWithDiscrimination()

    print(model)