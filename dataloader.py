from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader


def get_mnist_dataloaders(batch_size):
    '''
        Prepare MNIST DataLoaders for training, validation, and testing.
        args:
            batch_size (int): Number of samples per batch.
        return:
            train_loader, val_loader, test_loader: DataLoaders for MNIST dataset.
    '''
    # Define data transformation: convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3801,))
    ])

    # Load MNIST training and test datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Split training dataset into training and validation sets (90%/10%)
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataloader
    train_loader, val_loader, test_loader = get_mnist_dataloaders(batch_size=64)
    print("Number of training batches:", len(train_loader))
    print("Number of validation batches:", len(val_loader))
    print("Number of test batches:", len(test_loader))

    # Fetch one batch to check shapes
    images, labels = next(iter(train_loader))
    print("Batch images shape:", images.shape)  # [batch_size, 1, 28, 28]
    print("Batch labels shape:", labels.shape)  # [batch_size]

