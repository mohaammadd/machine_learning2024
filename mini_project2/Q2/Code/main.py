import os
from models import OneLayerNetwork
from config import configs
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils import plot_results, save_results_to_csv
from torchmetrics import Accuracy

# Define the AverageMeter class for tracking metrics
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Training loop for one epoch
def train_one_epoch(model, train_loader, criterion, optimizer, epoch=None):
    model.train()
    loss_train = 0
    accuracy_train = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(model.device), targets.to(model.device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        loss_train += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        accuracy_train += (predicted == targets).sum().item()

    return loss_train / len(train_loader), accuracy_train / total

# Evaluation loop
def evaluate(model, valid_loader, criterion):
    model.eval()
    loss_valid = AverageMeter()
    accuracy_valid = Accuracy(task="multiclass", num_classes=model.config["num_classes"]).to(model.device)

    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(model.device), targets.to(model.device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss_valid.update(loss.item(), inputs.size(0))
            accuracy_valid.update(outputs, targets)

    return loss_valid.avg, accuracy_valid.compute().item()

# Main function
def main(train_loader, valid_loader, num_epochs=500):
    os.makedirs("results", exist_ok=True)
    best_results = []
    last_epoch_results = []

    for config in configs:
        # Initialize Model and optimizer
        model = OneLayerNetwork(**config)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])

        train_loss_hist = []
        valid_loss_hist = []
        acc_train_hist = []
        acc_valid_hist = []

        best_epoch = None
        best_valid_loss = float('inf')

        for epoch in range(num_epochs):
            # Training step
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, epoch
            )
            # Validation step
            valid_loss, valid_acc = evaluate(model, valid_loader, criterion)

            train_loss_hist.append(train_loss)
            valid_loss_hist.append(valid_loss)
            acc_train_hist.append(train_acc)
            acc_valid_hist.append(valid_acc)

            # Check for best validation loss
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch + 1
                best_train_loss = train_loss
                best_train_acc = train_acc
                best_valid_acc = valid_acc

        # Log best results
        best_results.append({
            "Model": config["model_name"],
            "Activation": config["activation"],
            "Neurons": config["neurons"],
            "Best Epoch": best_epoch,
            "Train Loss": best_train_loss,
            "Train Accuracy": best_train_acc,
            "Valid Loss": best_valid_loss,
            "Valid Accuracy": best_valid_acc,
        })

        # Log last epoch results
        last_epoch_results.append({
            "Model": config["model_name"],
            "Activation": config["activation"],
            "Neurons": config["neurons"],
            "Last Epoch": num_epochs,
            "Train Loss": train_loss_hist[-1],
            "Train Accuracy": acc_train_hist[-1],
            "Valid Loss": valid_loss_hist[-1],
            "Valid Accuracy": acc_valid_hist[-1],
        })

        # Plot results for each model
        plot_results(
            train_loss_hist,
            valid_loss_hist,
            acc_train_hist,
            acc_valid_hist,
            config["model_name"],
            config["neurons"],
            config["activation"],
            save_path="results"
        )

    # Save results to CSV
    save_results_to_csv(best_results, "results/best_results.csv")
    save_results_to_csv(last_epoch_results, "results/last_epoch_results.csv")

# Data Preprocessing and Loading
data = pd.read_csv(r'C:\Users\MEHRAN\Desktop\HW2\project\data\teleCust1000t.csv')
df = pd.DataFrame(data)
x = df.drop(columns=['custcat'])
y = df['custcat']
y = y - 1  # Subtract 1 from all target values to make them zero-indexed

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_temp, y_train, y_temp = train_test_split(x_scaled, y, test_size=0.3, random_state=73)
x_valid, x_test, y_valid, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=73)

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

x_valid_tensor = torch.tensor(x_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid.values, dtype=torch.long)

x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Create TensorDataset
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
valid_dataset = TensorDataset(x_valid_tensor, y_valid_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Main program execution
if __name__ == "__main__":
    main(train_loader, valid_loader, num_epochs=600)
