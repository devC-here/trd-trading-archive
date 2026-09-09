import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

# 1. Load dataset
df = pd.read_csv("trade_decisions2.csv")

# 2. Map trade decisions to numerical values
decision_map = {"Sell": 0, "No Trade": 1, "Buy": 2}
df['trade_decision'] = df['trade_decision'].map(decision_map)

# 3. Select relevant features
features = [
    'open', 'close', 
    'high_low_high_close_ratio', 'high_close_low_close_ratio', 'open_low_high_close_ratio',
    'high_low_low_close_ratio', 'open_close_low_close_ratio',
    'wick_to_length_ratio', 'body_to_length_ratio', 'tail_to_length_ratio',
    'volume'
]
X = df[features].values
y = df['trade_decision'].values

# 4. Split data: 70% train, 15% validation, 15% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1765, random_state=42, stratify=y_train_val
)  # 0.1765 of 85% ≈ 15%

# 5. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# 6. Save the scaler for later use
joblib.dump(scaler, "scaler.pkl")

# 7. Convert to PyTorch tensors (for neural network training)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor   = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor   = torch.tensor(y_val, dtype=torch.long)
X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test, dtype=torch.long)

print("Dataset preprocessing complete! Ready for model training.")
