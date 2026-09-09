import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import joblib
import warnings
import logging
import time
from datetime import datetime

# Suppress all warnings from being printed to the console
warnings.filterwarnings("ignore")

# Import all predictor classes from the individual files
from adxmacdccideploy import MarketPredictor
from bollingerdeploy import TradingPredictor as BollingerTradingPredictor
from candledeploy import CandleSequencePredictor
from FRAMAdeploy import FRAMAPredictor
from positiondeploy import BollingerPredictor as PositionBollingerPredictor
from vwrsideploy import VWRSIPredictor

# Import the final decision function from fusion_engine
from fusion_engine import get_final_trading_signal

# Import trade placing function (assuming placer.py exists)
try:
    from placer import execute_automated_trade
except ImportError:
    print("Error: placer.py not found. Automated trading will be disabled.")
    execute_automated_trade = None # Set to None if import fails

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("trading_log.log"), # Log to a file
        logging.StreamHandler() # Also log to console
    ]
)

# --- Global Variables for Trade Management ---
active_trades = {} # To keep track of currently active trades by symbol

# --- Helper Function to Get Candle Data (for potential future use) ---
def get_candle_data(symbol, timeframe_code, count=2):
    """Fetches recent candle data from MetaTrader 5."""
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe_code, 0, count)
        if rates is None or len(rates) == 0:
            logging.warning(f"No data retrieved for {symbol} on timeframe {timeframe_code}")
            return None, None, None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        latest_high = df['high'].iloc[-1]
        latest_low = df['low'].iloc[-1]
        latest_close = df['close'].iloc[-1]
        return latest_high, latest_low, latest_close
    except Exception as e:
        logging.error(f"Error getting candle data for {symbol} ({timeframe_code}): {e}")
        return None, None, None


# --- Function to Process Predictions and Return Decision for a Given Timeframe ---
def get_timeframe_decision(timeframe_name, timeframe_code, predictors):
    """
    Collects predictions for a given timeframe and returns the final trading decision.
    Does NOT execute trades.
    """
    predictions = {}
    symbol = "XAUUSDm" # Assuming XAUUSDm is the primary symbol

    # Collect predictions
    if predictors.get('adx_cci_macd'):
        try:
            predictions['adx_cci_macd'] = predictors['adx_cci_macd'].get_prediction(symbol, timeframe_code)
        except Exception as e:
            logging.error(f"Error getting ADX/MACD/CCI prediction for {timeframe_name}: {e}")
            predictions['adx_cci_macd'] = {"error": f"Error: {e}"}

    if predictors.get('boll'):
        try:
            result = predictors['boll'].predict_timeframe(timeframe_name, timeframe_code)
            if "error" not in result:
                # Ensure 'predictions' key exists and is a dictionary
                if 'predictions' in result and isinstance(result['predictions'], dict):
                    predictions['boll'] = result['predictions']
                else: # Handle cases where 'predictions' might not be nested or is missing
                    predictions['boll'] = result # Assume result directly contains prediction data
            else:
                logging.error(f"Error in Bollinger prediction for {timeframe_name}: {result['error']}")
                predictions['boll'] = {"error": f"Error: {result['error']}"}
        except Exception as e:
            logging.error(f"Error getting Bollinger prediction for {timeframe_name}: {e}")
            predictions['boll'] = {"error": f"Error: {e}"}

    if predictors.get('candlestick'):
        try:
            predictions['candlestick'] = predictors['candlestick'].get_most_confident_trade(timeframe_code)
        except Exception as e:
            logging.error(f"Error getting Candlestick prediction for {timeframe_name}: {e}")
            predictions['candlestick'] = {"error": f"Error: {e}"}
            
    if predictors.get('frama'):
        try:
            frama_result = predictors['frama'].predict(timeframe_code)
            if "error" not in frama_result:
                predictions['frama'] = frama_result['predictions']
                predictions['frama']['timeframe'] = timeframe_name # Add timeframe name for consistency
            else:
                logging.error(f"Error in FRAMA prediction for {timeframe_name}: {frama_result['error']}")
                predictions['frama'] = {"timeframe": timeframe_name, "error": f"Error: {frama_result['error']}"}
        except Exception as e:
            logging.error(f"Error getting FRAMA prediction for {timeframe_name}: {e}")
            predictions['frama'] = {"timeframe": timeframe_name, "error": f"Error: {e}"}

    if predictors.get('position_bollinger'):
        try:
            result = predictors['position_bollinger'].predict(timeframe_code)
            if "error" not in result:
                predictions['position_bollinger'] = result
            else:
                logging.error(f"Error in Position Bollinger prediction for {timeframe_name}: {result['error']}")
                predictions['position_bollinger'] = {"error": f"Error: {result['error']}"}
        except Exception as e:
            logging.error(f"Error getting Position Bollinger prediction for {timeframe_name}: {e}")
            predictions['position_bollinger'] = {"error": f"Error: {e}"}

    if predictors.get('vwrsi'):
        try:
            vwrsi_result = predictors['vwrsi'].predict(timeframe_code)
            if "error" not in vwrsi_result:
                predictions['vwrsi'] = vwrsi_result # VWRSIPredictor returns a dictionary directly
                predictions['vwrsi']['timeframe'] = timeframe_name # Add timeframe name for consistency
            else:
                logging.error(f"Error in VW-RSI prediction for {timeframe_name}: {vwrsi_result['error']}")
                predictions['vwrsi'] = {"timeframe": timeframe_name, "error": f"Error: {vwrsi_result['error']}"}
        except Exception as e:
            logging.error(f"Error getting VW-RSI prediction for {timeframe_name}: {e}")
            predictions['vwrsi'] = {"timeframe": timeframe_name, "error": f"Error: {e}"}

    # Log predictions
    logging.info(f"--- {timeframe_name} Timeframe Predictions ---")
    if predictions:
        for predictor_name, result in predictions.items():
            logging.info(f"{predictor_name}: {result}")
    else:
        logging.info(f"No {timeframe_name} predictions available.")

    # Make Final Decision
    if predictions:
        final_dec = get_final_trading_signal(predictions)
        logging.info(f"--- {timeframe_name} Final Decision: {final_dec} ---")
        return {'decision': final_dec, 'timeframe': timeframe_name, 'symbol': symbol}
    else:
        logging.info(f"Not enough {timeframe_name} predictions to make a final decision. Defaulting to Hold.")
        return {'decision': 'Hold', 'timeframe': timeframe_name, 'symbol': symbol}


# --- Main Prediction and Decision Loop ---
def main():
    # Initialize MetaTrader 5
    if not mt5.initialize():
        logging.error("Failed to initialize MetaTrader 5. Exiting.")
        return # Exit if MT5 init fails

    # Initialize predictors only once
    adx_cci_macd_predictor = None
    bollinger_predictor = None
    candle_predictor = None
    frama_predictor = None
    position_bollinger_predictor = None
    vwrsi_predictor = None

    try:
        adx_cci_macd_predictor = MarketPredictor(symbols=["XAUUSDm"])
        logging.info("ADX/MACD/CCI Predictor initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize ADX/MACD/CCI Predictor: {e}")
        
    try:
        bollinger_predictor = BollingerTradingPredictor()
        logging.info("Bollinger Predictor initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize Bollinger Predictor: {e}")

    try:
        candle_predictor = CandleSequencePredictor(symbols=["XAUUSDm"])
        logging.info("Candle Sequence Predictor initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize Candle Sequence Predictor: {e}")

    try:
        frama_predictor = FRAMAPredictor("frama_multi_target_model.pth", "frama_seq_scaler.pkl")
        logging.info("FRAMA Predictor initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize FRAMA Predictor: {e}")
        
    try:
        position_bollinger_predictor = PositionBollingerPredictor()
        logging.info("Position Bollinger Predictor initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize Position Bollinger Predictor: {e}")

    try:
        vwrsi_predictor = VWRSIPredictor("vwrsi_multi_target_model.pth", "vwrsi_seq_scaler.pkl")
        logging.info("VW-RSI Predictor initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize VW-RSI Predictor: {e}")

    logging.info("Starting immediate prediction process for M30 and H1 timeframes...")

    # Get decisions for both M30 and H1
    m30_data = get_timeframe_decision("M30", mt5.TIMEFRAME_M30, {
        'adx_cci_macd': adx_cci_macd_predictor,
        'boll': bollinger_predictor,
        'candlestick': candle_predictor,
        'frama': frama_predictor,
        'position_bollinger': position_bollinger_predictor,
        'vwrsi': vwrsi_predictor
    })

    h1_data = get_timeframe_decision("H1", mt5.TIMEFRAME_H1, {
        # Note: ADX/MACD/CCI is often not used for H1 in this setup based on previous code.
        # Ensure correct predictors are passed for H1 based on your model's design.
        'boll': bollinger_predictor,
        'candlestick': candle_predictor,
        'frama': frama_predictor,
        'position_bollinger': position_bollinger_predictor,
        'vwrsi': vwrsi_predictor
    })

    final_symbol = "XAUUSDm"
    final_decision_to_execute = "Hold"
    timeframe_decision_came_from = "N/A"

    # Apply priority logic: H1 first, then M30 fallback
    if h1_data['decision'] in ["Buy", "Sell"]:
        final_decision_to_execute = h1_data['decision']
        timeframe_decision_came_from = "H1"
        logging.info(f"H1 decision ({final_decision_to_execute}) takes precedence.")
    elif m30_data['decision'] in ["Buy", "Sell"]:
        final_decision_to_execute = m30_data['decision']
        timeframe_decision_came_from = "M30"
        logging.info(f"H1 decision is 'Hold', falling back to M30 decision ({final_decision_to_execute}).")
    else:
        final_decision_to_execute = "Hold"
        timeframe_decision_came_from = "N/A"
        logging.info("Both H1 and M30 decisions are 'Hold'. No trade will be placed.")

    # --- Trade Execution Logic (now consolidated in main) ---
    if execute_automated_trade and (final_decision_to_execute == "Buy" or final_decision_to_execute == "Sell"):
        current_positions = mt5.positions_get(symbol=final_symbol)

        # Check if there's an ongoing trade for this symbol
        if current_positions:
            logging.info(f"Active positions found for {final_symbol}. Checking for conflicts.")
            current_trade_type = "Buy" if current_positions[0].type == mt5.ORDER_TYPE_BUY else "Sell"
            
            # If current decision conflicts with active trade, do not place
            if current_trade_type != final_decision_to_execute:
                logging.warning(f"Conflict: Current decision ({final_decision_to_execute}) from {timeframe_decision_came_from} differs from active trade ({current_trade_type}). Skipping trade.")
            else:
                logging.info(f"Decision ({final_decision_to_execute}) from {timeframe_decision_came_from} matches active trade. No new trade needed.")
        else:
            # No conflicting active trade, proceed to place
            logging.info(f"Attempting to place {final_decision_to_execute} trade for {final_symbol} based on {timeframe_decision_came_from} decision.")
            if execute_automated_trade(final_symbol, final_decision_to_execute):
                logging.info(f"Trade successful for {final_symbol} - {final_decision_to_execute} from {timeframe_decision_came_from}.")
                active_trades[final_symbol] = final_decision_to_execute
            else:
                logging.error(f"Trade failed for {final_symbol} - {final_decision_to_execute} from {timeframe_decision_came_from}.")
    elif not execute_automated_trade:
        logging.warning("placer.py or execute_automated_trade function not available. Automated trading skipped.")
    else:
        logging.info(f"Final combined decision is 'Hold'. No trade placed.")

# --- Entry Point ---
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("Prediction process interrupted by user (Ctrl+C). Shutting down.")
    except Exception as e:
        logging.critical(f"A critical error occurred in the main prediction process: {e}", exc_info=True)
    finally:
        # Shutdown MT5 and predictors gracefully
        logging.info("Shutting down MetaTrader5 and predictors.")
        try:
            mt5.shutdown()
        except Exception as e:
            logging.error(f"Error during MT5 shutdown: {e}") 