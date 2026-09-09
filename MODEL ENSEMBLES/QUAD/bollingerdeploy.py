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
            nn.Linear(hidden_dim, 3)  # Buy, Sell, No Trade
        )

    def forward(self, x):
        return self.model(x)

class TradingPredictor:
    def __init__(self):
        self.symbol = "XAUUSDm"
        self.timeframes = {
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1
        }

        self.scaler = joblib.load("bollinger_scaler.pkl")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TradeClassifier(input_dim=3).to(self.device)
        self.model.load_state_dict(torch.load("bollinger_trade_classifier.pth", map_location=self.device))
        self.model.eval()

        if not mt5.initialize():
            raise RuntimeError("MT5 Initialization Failed!")

    def extract_features(self, df):
        bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_middle"] = bb.bollinger_mavg()

        df["lb"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-6)
        df["ub"] = 1 - df["lb"]

        # Latest incursion tracing logic
        df["pi"] = 0.5  # Default middle line
        last_20 = df.iloc[-20:]
        for i in range(len(last_20) - 1, -1, -1):
            close = last_20.iloc[i]["close"]
            upper = last_20.iloc[i]["bb_upper"]
            lower = last_20.iloc[i]["bb_lower"]

            if close >= upper:
                df.at[last_20.index[-1], "pi"] = -1.0
                break
            elif close <= lower:
                df.at[last_20.index[-1], "pi"] = 1.0
                break
        return df.dropna()

    def predict_timeframe(self, timeframe_name, timeframe_code):
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe_code, 0, 50)
        if rates is None or len(rates) < 30:
            return {"decision": "No Trade", "confidence": 0.0, "timeframe": timeframe_name}

        df = pd.DataFrame(rates)
        df["close"] = df["close"].astype(float)
        df = self.extract_features(df)

        if df.empty:
            return {"decision": "No Trade", "confidence": 0.0, "timeframe": timeframe_name}

        latest = df.iloc[-1][["lb", "ub", "pi"]].values.reshape(1, -1)
        scaled = self.scaler.transform(latest)
        X_tensor = torch.tensor(scaled, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            output = self.model(X_tensor)
            prob = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_class = np.argmax(prob)
            confidence = prob[pred_class]

            # Manual mapping
            decision_map = {0: "Sell", 1: "No trade", 2: "Buy"}
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

        # Decision logic
        if decisions["M30"] == "No Trade" and decisions["H1"] == "No Trade":
            final_decision = "No Trade"
        elif decisions["H1"] != "No Trade" and decisions["M30"] == "No Trade":
            final_decision = decisions["H1"]
        elif decisions["M30"] != "No Trade" and decisions["H1"] == "No Trade":
            final_decision = decisions["M30"]
        elif decisions["M30"] == decisions["H1"]:
            final_decision = decisions["H1"]
        else:
            final_decision = decisions["H1"]  # Prefer H1 on conflict

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
