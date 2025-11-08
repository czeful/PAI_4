import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd

# -----------------------------
# 1. Подготовка данных
# -----------------------------
dataset_name = "MNIST"  

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

if dataset_name == "MNIST":
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
else:
    dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Разделяем train на train + validation
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# -----------------------------
# 2. Определение модели MLP
# -----------------------------
class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[256, 128, 64], output_size=10):
        super(MLP, self).__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

# -----------------------------
# 3. Гиперпараметры экспериментов
# -----------------------------
epochs_list = [5, 10, 20]
batch_sizes = [32, 64, 128]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = []

# -----------------------------
# 4. Обучение и оценка
# -----------------------------
for epochs, batch_size in itertools.product(epochs_list, batch_sizes):
    print(f"Training: epochs={epochs}, batch_size={batch_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        # Training
        model.train()
        correct_train, total_train, loss_train = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            loss_train += loss.item() * x.size(0)
            _, predicted = torch.max(output, 1)
            total_train += y.size(0)
            correct_train += (predicted == y).sum().item()
        train_losses.append(loss_train/total_train)
        train_accs.append(correct_train/total_train)
        
        # Validation
        model.eval()
        correct_val, total_val, loss_val = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                loss_val += loss.item() * x.size(0)
                _, predicted = torch.max(output, 1)
                total_val += y.size(0)
                correct_val += (predicted == y).sum().item()
        val_losses.append(loss_val/total_val)
        val_accs.append(correct_val/total_val)

    results.append({
        "epochs": epochs,
        "batch_size": batch_size,
        "train_acc": train_accs[-1],
        "val_acc": val_accs[-1],
        "train_loss": train_losses[-1],
        "val_loss": val_losses[-1]
    })

# -----------------------------
# 5. Вывод результатов
# -----------------------------
df_results = pd.DataFrame(results)
print("\nResults table:")
print(df_results)

# -----------------------------
# 6. Визуализация предсказаний
# -----------------------------
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
model.eval()
all_images, all_labels, all_preds = [], [], []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        _, predicted = torch.max(output, 1)
        all_images.append(x.cpu())
        all_labels.append(y.cpu())
        all_preds.append(predicted.cpu())

images = torch.cat(all_images)
labels = torch.cat(all_labels)
preds = torch.cat(all_preds)

correct_idx = (labels == preds).nonzero().squeeze()
incorrect_idx = (labels != preds).nonzero().squeeze()

def plot_examples(indices, title):
    plt.figure(figsize=(12,3))
    for i, idx in enumerate(indices[:5]):
        plt.subplot(1,5,i+1)
        plt.imshow(images[idx].view(28,28), cmap='gray')
        plt.title(f"T:{labels[idx].item()} P:{preds[idx].item()}")
        plt.axis('off')
    plt.suptitle(title)
    plt.show()

plot_examples(correct_idx, "Correct Predictions")
plot_examples(incorrect_idx, "Incorrect Predictions")
