import torch
from dataloader import get_mnist_dataloaders
from model import LeNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
_, _, test_loader = get_mnist_dataloaders(batch_size=64)

# Load trained model
model = LeNet().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = correct / total
print(f"Test Accuracy: {test_acc:.4f}")
