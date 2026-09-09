import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib

# Define the model architecture (must match training architecture)
class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 3)
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)  # Reshape for LSTM
        x, _ = self.lstm(x)
        x = self.leaky_relu(self.fc1(x[:, -1, :]))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the trained model
model = CNN_LSTM()
model.load_state_dict(torch.load("D:/projects/RESEARCH.py/cnn_lstm_xauusd_model.pth"))
model.eval()

# Load the scaler
scaler = joblib.load("scaler.pkl")

# Load unseen data
df = pd.read_csv("trade_decisions2.csv")

# Rename the column to match expected format
df.rename(columns={"Trade_Decision": "trade_decision"}, inplace=True)

decision_map = {"Sell": 0, "No Trade": 1, "Buy": 2}
df['trade_decision'] = df['trade_decision'].map(decision_map)

# Extract OHLCV features
features = ['open', 'high', 'low', 'close', 'volume']
X_unseen = df[features].values
y_true = df['trade_decision'].values

# Scale features
X_unseen = scaler.transform(X_unseen)
X_unseen = X_unseen.reshape((X_unseen.shape[0], 1, X_unseen.shape[1]))

# Convert to tensor
X_unseen_tensor = torch.tensor(X_unseen, dtype=torch.float32)

# Get predictions
with torch.no_grad():
    outputs = model(X_unseen_tensor)
    predictions = torch.argmax(outputs, dim=1).numpy()

# Calculate accuracy
accuracy = np.mean(predictions == y_true) * 100
print(f"Model Accuracy on Unseen Data: {accuracy:.2f}%")
