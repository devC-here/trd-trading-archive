from deploy import TradingPredictor
from datetime import datetime
import MetaTrader5 as mt5

def test_trading_predictor():
    print("\n🔍 Running TradingPredictor Test...")
    
    # Initialize predictor
    try:
        predictor = TradingPredictor()
        print("✅ TradingPredictor initialized successfully.")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return

    # Test model loading
    if predictor.model:
        print("✅ Model loaded successfully.")
    else:
        print("❌ Model failed to load.")

    # Test scaler and label encoder
    if predictor.scaler and predictor.label_encoder:
        print("✅ Scaler and Label Encoder loaded successfully.")
    else:
        print("❌ Scaler or Label Encoder failed to load.")

    # Check if MetaTrader 5 is initialized
    if mt5.terminal_info() is None:
        print("❌ MT5 initialization failed!")
        return
    print("✅ MT5 initialized successfully.")

    # Test predictions for all symbols
    timeframe = mt5.TIMEFRAME_M15  # Example timeframe
    predictions = []
    for symbol in predictor.symbols:
        try:
            prediction = predictor.get_prediction(symbol, timeframe)
            predictions.append(prediction)
            print(f"🔹 Prediction for {symbol}: {prediction}")
        except Exception as e:
            print(f"❌ Error predicting {symbol}: {e}")

    # Find the most confident prediction
    try:
        best_trade = predictor.get_most_confident_prediction(timeframe)
        print(f"\n✅ Most Confident Trade: {best_trade}")
    except Exception as e:
        print(f"❌ Error selecting best trade: {e}")

if __name__ == "__main__":
    test_trading_predictor()
