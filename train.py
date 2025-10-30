import torch
from dataloader import get_mnist_dataloaders
from model import LeNet
import torch.nn as nn
import torch.optim as optim
import os
import csv
from tqdm import tqdm
import argparse


def evaluate(model, loader, device):
    '''
    Evaluate model accuracy on given DataLoader.

    args:
        model: LeNet
        loader: DataLoader providing data
        device: 'cuda' or 'cpu'
    
    returns:
        accuracy: float, fraction of correctly predicted samples
    '''
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted ==labels).sum().item()
    return correct / total


def train(args):
    '''
    Train the LeNet model on MNIST dataset with given hyperparameters.
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, val_loader, test_loader = get_mnist_dataloaders(args.batch_size)

    # Load MNIST data loaders
    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    # Prepare CSV logging directory
    if not os.path.exists('logs'):
        os.makedirs('logs')
    csv_file = open('logs/training_log.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'val_acc'])

    best_val_acc = 0.0

    # Training loop
    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset) # type: ignore

        # Evaluate on validation set
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch [{epoch}/{args.epochs}] - Loss: {epoch_loss:.4f} - Val Acc: {val_acc:.4f}")
        csv_writer.writerow([epoch, epoch_loss, val_acc])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_lenet5_mnist.pth")
            print("Saved best model!")

    csv_file.close()
    # Evaluate model on test set after training
    test_acc = evaluate(model, test_loader, device)
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Strict LeNet-5 on MNIST")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--weight_decay', type =float, default= 1e-4)
    args = parser.parse_args()
    train(args)
