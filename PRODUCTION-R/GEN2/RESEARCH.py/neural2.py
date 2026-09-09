import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Load dataset
df = pd.read_csv("trade_decisions2.csv")  # Ensure this file exists

decision_map = {"Sell": 0, "No Trade": 1, "Buy": 2}
df['trade_decision'] = df['trade_decision'].map(decision_map)

# Use only OHLCV features
features = ['open', 'high', 'low', 'close', 'volume']
X = df[features].values
y = df['trade_decision'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")  # Save scaler

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class EnhancedNeuralNetwork(nn.Module):
    def __init__(self):
        super(EnhancedNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 3)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(0.3)
        
        self.attn_weight = nn.Linear(256, 1)  # Apply attention at fc3 output
    
    def forward(self, x):
        x1 = self.leaky_relu(self.bn1(self.fc1(x)))
        x1 = self.dropout(x1)
        x2 = self.leaky_relu(self.bn2(self.fc2(x1)))
        x2 = self.dropout(x2)
        x3 = self.leaky_relu(self.bn3(self.fc3(x2)))

        # Apply self-attention after fc3
        attn_scores = torch.softmax(self.attn_weight(x3), dim=1)
        x3 = x3 * attn_scores

        x4 = self.leaky_relu(self.fc4(x3))
        x5 = self.fc5(x4)
        return x5

model = EnhancedNeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3)

train_losses = []
test_accuracies = []
epochs = 250
early_stopping_patience = 15
best_loss = float('inf')
best_acc = 0.0
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions.eq(y_test_tensor).sum() / y_test_tensor.shape[0]).item()
        test_accuracies.append(accuracy)
    
    scheduler.step(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")
    
    if avg_loss < best_loss or accuracy > best_acc:
        best_loss = avg_loss
        best_acc = accuracy
        best_model_state = model.state_dict()
        no_improvement_epochs = 0
    else:
        no_improvement_epochs += 1
        if no_improvement_epochs >= early_stopping_patience:
            print("Early stopping triggered.")
            break

if best_model_state:
    model.load_state_dict(best_model_state)

torch.save(model.state_dict(), "xauusd_model.pth")
print("Best model saved as xauusd_model.pth")

log_df = pd.DataFrame({"epoch": list(range(1, len(train_losses) + 1)),
                       "train_loss": train_losses,
                       "test_accuracy": [x * 100 for x in test_accuracies]})
log_df.to_csv("training_log.csv", index=False)
print("Training log saved as training_log.csv")

traced_model = torch.jit.trace(model, torch.rand(1, X_train.shape[1]))
torch.jit.save(traced_model, "xauusd_model_traced.pth")
print("TorchScript model saved as xauusd_model_traced.pth")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(test_accuracies) + 1), [x * 100 for x in test_accuracies], label='Test Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy Curve')
plt.legend()
plt.grid()

plt.show()
