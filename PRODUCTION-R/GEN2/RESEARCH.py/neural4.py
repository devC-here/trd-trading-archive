import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Load dataset
df = pd.read_csv("trade_decisions1.csv")  # Ensure this file exists

# Map trade decisions to numerical values
decision_map = {"Sell": 0, "No Trade": 1, "Buy": 2}
df['trade_decision'] = df['trade_decision'].map(decision_map)

# Compute ratio-based features
df['open_high_ratio'] = df['open'] / df['high']
df['open_low_ratio'] = df['open'] / df['low']
df['open_close_ratio'] = df['open'] / df['close']
df['high_low_ratio'] = df['high'] / df['low']
df['high_close_ratio'] = df['high'] / df['close']
df['low_close_ratio'] = df['low'] / df['close']

# Select features dynamically (excluding time and trade_decision)
features = [col for col in df.columns if col not in ['time', 'trade_decision']]
X = df[features].values
y = df['trade_decision'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features and save scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")  # Save scaler for inference

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
batch_size = 128  # Increased batch size for stability
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define Neural Network
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 3)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.2)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(128)
    
    def forward(self, x):
        x = self.leaky_relu(self.norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.norm3(self.fc3(x)))
        x = self.leaky_relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Initialize model, loss, and optimizer
model = NeuralNetwork(input_dim=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0001)

# Train the model with early stopping
epochs = 200
early_stopping_patience = 10
best_loss = float('inf')
no_improvement_epochs = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Prevent exploding gradients
        optimizer.step()
        total_loss += loss.item()
    
    scheduler.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

    # Early Stopping
    if total_loss < best_loss:
        best_loss = total_loss
        no_improvement_epochs = 0
    else:
        no_improvement_epochs += 1
        if no_improvement_epochs >= early_stopping_patience:
            print("Early stopping triggered.")
            break

# Evaluate model
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    predictions = torch.argmax(outputs, dim=1)
    accuracy = (predictions.eq(y_test_tensor).sum() / y_test_tensor.shape[0]).item()
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Feature Importance Calculation
feature_importance = torch.abs(model.fc1.weight).mean(dim=0).detach().cpu().numpy()
feature_importance_dict = {features[i]: feature_importance[i] for i in range(len(features))}
feature_importance_sorted = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
print("Feature Importance Ranking:")
for feature, importance in feature_importance_sorted:
    print(f"{feature}: {importance:.4f}")

# Print least important feature
least_important_feature = feature_importance_sorted[-1]
print(f"Least Important Feature: {least_important_feature[0]}: {least_important_feature[1]:.4f}")

# Save model
torch.save(model.state_dict(), "xauusd_model.pth")
print("Model saved as xauusd_model.pth")
