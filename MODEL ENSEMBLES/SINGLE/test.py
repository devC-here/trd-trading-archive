import MetaTrader5 as mt5
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import logging

# ================= Logging =================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ================= Neural Network =================
class TradeDecisionNN(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3):
        super(TradeDecisionNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 3)  # Buy, Sell, No Trade
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        return self.fc4(x)

# ================= Feature Engineering =================
def compute_features(df):
    wick = df['high'] - df[['open','close']].max(axis=1)
    body = (df['open'] - df['close']).abs()
    tail = df[['open','close']].min(axis=1) - df['low']
    candle_len = (df['high'] - df['low']).replace(0, 1e-6)

    df['wick_to_length_ratio'] = wick / candle_len
    df['body_to_length_ratio'] = body / candle_len
    df['tail_to_length_ratio'] = tail / candle_len
    return df

# ================= Backtesting Function =================
def backtest(symbol="EURUSDm", timeframe=mt5.TIMEFRAME_M15, n=2000, lookahead=3):
    logging.info("MT5 initialized.")

    # Load scaler and label encoder
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    # Load trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TradeDecisionNN(input_dim=3).to(device)
    model.load_state_dict(torch.load("xauusd_model_best.pth", map_location=device))
    model.eval()

    # Fetch historical rates
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, n)
    df = pd.DataFrame(rates)
    df = compute_features(df)

    feature_cols = ['wick_to_length_ratio', 'body_to_length_ratio', 'tail_to_length_ratio']
    features = scaler.transform(df[feature_cols])
    X = torch.tensor(features, dtype=torch.float32).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(X)
        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        decisions = label_encoder.inverse_transform(preds)

    # Initialize metrics
    total, correct, sl_hits, failed, score = 0, 0, 0, 0, 0.0

    for i in range(len(df) - lookahead):
        decision = decisions[i]
        if decision not in ["Buy", "Sell"]:
            continue

        total += 1
        curr_candle = df.iloc[i]
        open_pred = curr_candle['open']
        sl = curr_candle['low'] if decision == "Buy" else curr_candle['high']

        outcome_recorded = False

        # Sequential check next candles
        for j in range(1, lookahead + 1):
            next_c = df.iloc[i + j]
            open_next, close_next = next_c['open'], next_c['close']
            high_next, low_next = next_c['high'], next_c['low']

            # Check profit
            if decision == "Buy" and close_next > open_pred:
                correct += 1
                score += close_next - open_pred
                outcome_recorded = True
            elif decision == "Sell" and close_next < open_pred:
                correct += 1
                score += open_pred - close_next
                outcome_recorded = True

            # Check SL hit
            if not outcome_recorded:
                if decision == "Buy" and low_next <= sl:
                    sl_hits += 1
                    score += sl - open_pred  # negative
                    outcome_recorded = True
                elif decision == "Sell" and high_next >= sl:
                    sl_hits += 1
                    score += open_pred - sl  # negative
                    outcome_recorded = True

            if outcome_recorded:
                break  # Stop checking further candles

        # If neither profit nor SL hit
        if not outcome_recorded:
            failed += 1

    accuracy = correct / total if total > 0 else 0.0
    sl_hit_rate = sl_hits / total if total > 0 else 0.0

    # Print results
    print("============== Backtest Results ==============")
    print(f"Symbol        : {symbol}")
    print(f"Timeframe     : 15M")
    print(f"Samples Tested: {n}")
    print(f"Trades Taken  : {total}")
    print(f"Correct Trades: {correct}")
    print(f"SL Hits       : {sl_hits}")
    print(f"Failed Trades : {failed}")
    print(f"Accuracy      : {accuracy:.2%}")
    print(f"SL Hit Rate   : {sl_hit_rate:.2%}")
    print(f"Total Score   : {score:.5f} (positive=profit, negative=SL)")
    print(f"Lookahead     : next {lookahead} candles (sequential check)")
    print("==============================================")

# ================= Main =================
def main():
    if not mt5.initialize():
        logging.error("MT5 Initialization Failed!")
        return
    try:
        backtest("EURUSDm", mt5.TIMEFRAME_M30, n=4000, lookahead=10)
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
