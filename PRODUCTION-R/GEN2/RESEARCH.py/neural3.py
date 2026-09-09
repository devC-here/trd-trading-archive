import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("trade_decisions1.csv")

# Map trade decisions to numerical values
decision_map = {"Sell": 0, "No Trade": 1, "Buy": 2}
df['trade_decision'] = df['trade_decision'].map(decision_map)

# Use only OHLCV features
features = ['open', 'high', 'low', 'close', 'volume']
X = df[features].values
y = df['trade_decision'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")
print("Scaler saved as scaler.pkl")

# Reshape data for CNN+LSTM (adding channel dimension)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# DataLoader
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# CNN + LSTM Model
class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=2, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.lstm = nn.LSTM(input_size=64, hidden_size=256, num_layers=3, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 3)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.leaky_relu(self.fc1(x[:, -1, :]))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initialize model, loss, optimizer
model = CNN_LSTM()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training Loop
epochs = 200
best_loss = float('inf')
best_acc = 0.0
best_model_state = None

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)

    # Evaluate model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions.eq(y_test_tensor).sum() / y_test_tensor.shape[0]).item()
    
    # Track best model
    if avg_loss < best_loss or accuracy > best_acc:
        best_loss = avg_loss
        best_acc = accuracy
        best_model_state = model.state_dict()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

# Save best model
torch.save(best_model_state, "cnn_lstm_xauusd_model.pth")
print("Best model saved as cnn_lstm_xauusd_model.pth")
