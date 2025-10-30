import torch.nn as nn
import torch
from torchinfo import summary


class LeNet(nn.Module):
    '''
    LeNet CNN for MNIST classification.
    Input: 1x28x28 grayscale image
    Output: 10-class probabilities
    '''
    def __init__(self):
        super(LeNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0) # 28x28 -> 24x24
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2) # 24x24 -> 12x12

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0) # 12x12 -> 8x8
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2) # 8x8 ->4x4

        self.conv3 = nn.Conv2d(16, 120, kernel_size=4, stride=1, padding=0) # 4x4 -> 1x1

        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
    
    def forward(self,x):
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = torch.tanh(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Instantiate the model and move to device
    model = LeNet().to(device)
    print(model)

    # Print a detailed summary of the model (layers, output shapes, parameters)
    summary(model, input_size=(64,1,28,28), device=str(device))

    # Test forward pass with random input tensor
    x = torch.randn(64, 1, 28, 28).to(device)
    y = model(x)
    print("Output shape:", y.shape)