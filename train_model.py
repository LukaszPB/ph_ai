import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt

def train_and_evaluate_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=50):
    train_losses = []
    val_losses = []
    train_r2_scores = []
    val_r2_scores = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        y_train_true = []
        y_train_pred = []
        y_val_true = []
        y_val_pred = []

        with torch.no_grad():
            for inputs, targets in train_loader:
                outputs = model(inputs)
                y_train_true.extend(targets.numpy())
                y_train_pred.extend(outputs.numpy())

            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                y_val_true.extend(targets.numpy())
                y_val_pred.extend(outputs.numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        train_r2 = r2_score(y_train_true, y_train_pred)
        val_r2 = r2_score(y_val_true, y_val_pred)
        train_r2_scores.append(train_r2)
        val_r2_scores.append(val_r2)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, Train R2: {train_r2}, Val R2: {val_r2}')
    
    with mlflow.start_run():
        mlflow.log_metric("train_loss", train_losses[-1])
        mlflow.log_metric("val_loss", val_losses[-1])
        mlflow.log_metric("train_r2", train_r2_scores[-1])
        mlflow.log_metric("val_r2", val_r2_scores[-1])

        mlflow.pytorch.log_model(model, "model")

        plt.figure()
        plt.plot(range(num_epochs), train_losses, label="Train Loss")
        plt.plot(range(num_epochs), val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Curve")
        plt.savefig("loss_curve.png")
        mlflow.log_artifact("loss_curve.png")

        plt.figure()
        plt.plot(range(num_epochs), train_r2_scores, label="Train R2 Score")
        plt.plot(range(num_epochs), val_r2_scores, label="Validation R2 Score")
        plt.xlabel("Epoch")
        plt.ylabel("R2 Score")
        plt.legend()
        plt.title("Learning Curve")
        plt.savefig("learning_curve.png")
        mlflow.log_artifact("learning_curve.png")

# Wczytanie sformatowanych danych z pliku CSV
df = pd.read_csv('failures_data_formatted_merged.csv')

# Podział danych na cechy (X) i zmienną docelową (y)
X = df.drop('POTENTIAL_PRICE', axis=1)
y = df['POTENTIAL_PRICE']

# Podział danych na zestawy treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Konwersja danych do tensorów
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Stworzenie DataLoaderów
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Definicja sieci neuronowej
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Inicjalizacja modelu, funkcji straty i optymalizatora
model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Trenowanie i ewaluacja modelu oraz zapis w MLflow
train_and_evaluate_model(model, criterion, optimizer, train_loader, test_loader)