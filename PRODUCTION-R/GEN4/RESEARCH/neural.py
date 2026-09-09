import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

# Load dataset
df = pd.read_csv("trade_decisions2.csv")  # Ensure this file exists

# Map trade decisions to numerical values
decision_map = {"Sell": 0, "No Trade": 1, "Buy": 2}
df['trade_decision'] = df['trade_decision'].map(decision_map)

# Compute financial ratios
features = [
    'high_low_high_close_ratio', 'high_close_low_close_ratio',
    'open_low_high_close_ratio', 'high_low_low_close_ratio',
    'open_close_low_close_ratio', 'wick_to_length_ratio',
    'body_to_length_ratio', 'tail_to_length_ratio', 'volume'
]
X = df[features].values
y = df['trade_decision'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")  # Save scaler

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 3)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.leaky_relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Initialize model, loss, and optimizer
model = NeuralNetwork()
weights = torch.tensor([0.75, 1.0, 1.25], dtype=torch.float32)  # Adjust class balance
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.AdamW(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5)

# Train model with early stopping
epochs = 200
early_stopping_patience = 10
best_loss = float('inf')
no_improvement_epochs = 0
best_model_state = None

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    scheduler.step(total_loss / len(train_loader))

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")
    
    # Early Stopping
    if total_loss < best_loss:
        best_loss = total_loss
        best_model_state = model.state_dict()
        no_improvement_epochs = 0
    else:
        no_improvement_epochs += 1
        if no_improvement_epochs >= early_stopping_patience:
            print("Early stopping triggered.")
            break

# Save best model
if best_model_state:
    torch.save(best_model_state, "xauusd_model_best.pth")
    print("Best model saved as xauusd_model_best.pth")

# Evaluate model
model.load_state_dict(torch.load("xauusd_model_best.pth"))  # Load best model
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    predictions = torch.argmax(outputs, dim=1)
    accuracy = (predictions.eq(y_test_tensor).sum() / y_test_tensor.shape[0]).item()
    print(f"Test Accuracy: {accuracy * 100:.2f}%")