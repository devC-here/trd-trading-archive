import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib

# ✅ Define CNN-LSTM Model (Without LayerNorm)
class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        
        # ✅ Increase hidden_size from 128 → 256
        self.lstm = nn.LSTM(input_size=64, hidden_size=256, num_layers=2, batch_first=True, dropout=0.1)
        
        # ✅ Adjust fully connected layers
        self.fc1 = nn.Linear(256, 128)  # Increased due to larger hidden_size
        self.fc2 = nn.Linear(128, 3)
        
        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)  # ✅ Reduced dropout (better for small data)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)  # Reshape for LSTM
        x, _ = self.lstm(x)
        x = self.leaky_relu(self.fc1(x[:, -1, :]))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ✅ Load Model (Ignore Missing Keys)
model = CNN_LSTM()
try:
    checkpoint_path = "D:/projects/RESEARCH.py/cnn_lstm_xauusd_model.pth"
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model Loading Error: {e}")

# ✅ Load Scaler
try:
    scaler = joblib.load("D:/projects/RESEARCH.py/scaler.pkl")
    print("✅ Scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading scaler: {e}")

# ✅ Load Data
df = pd.read_csv("D:/projects/RESEARCH.py/trade_decisions2.csv")
df.columns = df.columns.str.lower()  # Normalize column names

features = ['open', 'high', 'low', 'close', 'volume']
if not all(col in df.columns for col in features + ['trade_decision']):
    raise ValueError("Dataset must contain OHLCV and 'trade_decision' columns!")

# ✅ Apply Data Augmentation (Random Scaling)
df[features] = df[features] * (1 + np.random.uniform(-0.01, 0.01, df[features].shape))

# ✅ Map trade decisions
decision_map = {"Sell": 0, "No Trade": 1, "Buy": 2}
df['trade_decision'] = df['trade_decision'].map(decision_map)

# ✅ Extract Features & Labels
X_unseen = df[features].values
y_true = df['trade_decision'].values

# ✅ Scale Features
X_unseen = scaler.transform(X_unseen)
X_unseen = X_unseen.reshape((X_unseen.shape[0], 1, X_unseen.shape[1]))

# ✅ Convert to Tensor
X_unseen_tensor = torch.tensor(X_unseen, dtype=torch.float32)

# ✅ Run Predictions
with torch.no_grad():
    outputs = model(X_unseen_tensor)
    predictions = torch.argmax(outputs, dim=1).numpy()

# ✅ Calculate Accuracy
accuracy = np.mean(predictions == y_true) * 100
print(f"🚀 Enhanced Model Accuracy on Unseen Data: {accuracy:.2f}%")
