import MetaTrader5 as mt5
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
import logging
import ta

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TradeClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.2):
        super(TradeClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)  # Output: [Sell, No Trade, Buy]
        )

    def forward(self, x):
        return self.model(x)

class TradingPredictor:
    def __init__(self):
        self.symbol = "BTCUSDm"
        self.timeframes = {
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1
        }

        self.scaler = joblib.load("adx_scaler.pkl")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TradeClassifier(input_dim=3).to(self.device)
        self.model.load_state_dict(torch.load("adx_trade_classifier.pth", map_location=self.device))
        self.model.eval()

        if not mt5.initialize():
            raise RuntimeError("MT5 Initialization Failed!")

    def extract_adx_features(self, df):
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)

        adx_indicator = ta.trend.ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
        df["adx"] = adx_indicator.adx()
        df["plus_di"] = adx_indicator.adx_pos()
        df["minus_di"] = adx_indicator.adx_neg()
        return df.dropna()

    def predict_timeframe(self, timeframe_name, timeframe_code):
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe_code, 0, 100)
        if rates is None or len(rates) < 30:
            return {"decision": "No Trade", "confidence": 0.0, "timeframe": timeframe_name}

        df = pd.DataFrame(rates)
        df = self.extract_adx_features(df)

        if df.empty:
            return {"decision": "No Trade", "confidence": 0.0, "timeframe": timeframe_name}

        latest = df.iloc[-1][["adx", "plus_di", "minus_di"]].values.reshape(1, -1)

        # Create a DataFrame with proper feature names before scaling
        feature_names = ["adx", "plus_di", "minus_di"]
        latest_df = pd.DataFrame(latest, columns=feature_names)

        # Apply the scaler
        scaled = self.scaler.transform(latest_df)

        X_tensor = torch.tensor(scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            output = self.model(X_tensor)
            prob = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_class = np.argmax(prob)
            confidence = prob[pred_class]

        decision_map = {0: "Sell", 1: "No Trade", 2: "Buy"}
        decision = decision_map[pred_class]

        logging.info(f"{self.symbol} [{timeframe_name}] → {decision} ({confidence:.2%})")
        return {
            "decision": decision,
            "confidence": confidence,
            "timeframe": timeframe_name
        }

    def get_final_decision(self):
        m30_result = self.predict_timeframe("M30", self.timeframes["M30"])
        h1_result = self.predict_timeframe("H1", self.timeframes["H1"])

        decisions = {
            "M30": m30_result["decision"],
            "H1": h1_result["decision"]
        }

        if decisions["M30"] == "No Trade" and decisions["H1"] == "No Trade":
            final_decision = "No Trade"
        elif decisions["H1"] != "No Trade" and decisions["M30"] == "No Trade":
            final_decision = decisions["H1"]
        elif decisions["M30"] != "No Trade" and decisions["H1"] == "No Trade":
            final_decision = decisions["M30"]
        elif decisions["M30"] == decisions["H1"]:
            final_decision = decisions["H1"]
        else:
            final_decision = decisions["H1"]  # Prefer H1 in conflict

        logging.info(f"FINAL DECISION: {final_decision}")
        return {
            "final_decision": final_decision,
            "details": {
                "M30": m30_result,
                "H1": h1_result
            }
        }

    def shutdown(self):
        mt5.shutdown()
        logging.info("MT5 shutdown.")

if __name__ == "__main__":
    predictor = TradingPredictor()
    decision_result = predictor.get_final_decision()

    print("\n=== FINAL TRADE DECISION ===")
    print("Decision:", decision_result["final_decision"])
    print("M30:", decision_result["details"]["M30"])
    print("H1: ", decision_result["details"]["H1"])

    predictor.shutdown()
