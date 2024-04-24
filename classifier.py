import argparse
parser = argparse.ArgumentParser(description="Train a ResNet model for binary classification of images.")
parser.add_argument('--data_dir', type=str, required=True, help='Directory with the dataset.')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training and testing.')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
parser.add_argument('--local', type=str, default='', help='Directory with the dataset.')
parser.add_argument('--job_id', type=str, default='local', help='Directory with the dataset.')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--mode', type=str, default='train')

args = parser.parse_args()

import os
if args.local != '':
    os.environ['CUDA_VISIBLE_DEVICES'] = args.local

import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from tqdm import tqdm

def set_seed(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(args.seed)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

if args.mode == "train":

    # Load datasets
    train_dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=args.data_dir.replace("classifier_data_7_7", "classifier_test_7_7"), transform=test_transform)
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    # Load a pre-trained ResNet and modify it
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # '2' for binary classification
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-5)

    best_correct = 0

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        batch_counter = 0
        correct = 0
        total = 0
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            batch_counter += 1

            running_loss += loss.item()

            train_bar.set_description(f"Epoch: [{epoch+1}/{args.epochs}] Loss: [{(running_loss / batch_counter):.04f}] Accuracy: {(100 * correct / total):.02f}%")

        # print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

        if (epoch + 1) % 5 == 0:

            # Evaluate the model
            model.eval()
            correct = 0
            total = 0
            test_bar = tqdm(test_loader)
            with torch.no_grad():
                for inputs, labels in test_bar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    test_bar.set_description(f"Test Epoch: [{epoch+1}/{args.epochs}] Accuracy: {(100 * correct / total):.02f}%")

            if correct > best_correct:
                best_correct = correct
                torch.save(model.state_dict(), "./results/best_classifier.pt")
                print("saved")

elif args.mode == "test":

# def load_and_test_model(model_path, data_dir, batch_size=32):
    model_path = "./results/best_classifier.pt"
    
    dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Model setup
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # Test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Tested Model Accuracy: {100 * correct / total}%')
