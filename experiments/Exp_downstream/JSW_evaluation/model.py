
import torch
import torch.nn as nn
import torchvision.models as models

class JSWRegression(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(JSWRegression, self).__init__()
        # Load a pretrained ResNet model
        self.resnet = models.resnet50(pretrained=False)

        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        
        # Remove the final fully connected layer (classification layer)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Add a new regression head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, out_channels)
        )

    def forward(self, x):
        # Extract features using ResNet
        features = self.resnet(x)
        # Pass features through the regression head
        output = self.fc(features)
        return output