import torch
import torch.nn as nn

class ConvAutoencoderWithClassification(nn.Module):
    def __init__(self):
        super(ConvAutoencoderWithClassification, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Normalize output to [0, 1] range
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),  # Binary classification (good vs anomaly)
            nn.Sigmoid()  # Output probability
        )

    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)
        
        # Decoder
        reconstructed = self.decoder(encoded)
        
        # Classification
        global_features = torch.mean(encoded, dim=[2, 3])  # Global Average Pooling
        class_output = self.classifier(global_features)
        
        return reconstructed, class_output
