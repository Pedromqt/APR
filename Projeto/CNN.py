import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNN, self).__init__()
        
        # Camadas Convolucionais
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Camadas FC 
        self.fc1 = nn.Linear(128*3*3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # pooling e ReLU na primeira camada convolucional
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        # pooling e ReLU na segunda camada convolucional
        x= F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        # pooling e ReLU na terceira camada convolucional
        x= F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        # flatten
        x = x.view(x.size(0), -1)

        # Camadas FC
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x