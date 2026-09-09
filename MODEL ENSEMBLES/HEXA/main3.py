import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import joblib
import warnings
import logging
import time
import json
from datetime import datetime, timedelta # Added timedelta for time calculations

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
# Configure logging to file only for detailed output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("trading_log.log") # Only log to file
    ]
)

# --- Global Variables for Trade Management and Timing ---
active_trades = {} # To keep track of currently active trades by symbol
last_main_run_minute_interval = -1 # To track when main logic last ran for a specific 30-min interval

# Global dictionary to hold initialized predictor instances
all_predictors = {}

# --- Helper Function to Initialize All Predictors (adapted from observer) ---
def initialize_all_predictors_main():
    """
    Initializes all market predictor models.
    Stores initialized predictor instances in the global 'all_predictors' dictionary.
    """
    global all_predictors
    # Define a dictionary mapping predictor names to their class and constructor arguments
    configs = {
        'adx_cci_macd': {'cls': MarketPredictor, 'args': {"symbols": ["XAUUSDm"]}},
        'boll': {'cls': BollingerTradingPredictor, 'args': {}},
        'candlestick': {'cls': CandleSequencePredictor, 'args': {"symbols": ["XAUUSDm"]}},
        'frama': {'cls': FRAMAPredictor, 'args': {"model_path": "frama_multi_target_model.pth", "scaler_path": "frama_seq_scaler.pkl"}},
        'position_bollinger': {'cls': PositionBollingerPredictor, 'args': {}},
        'vwrsi': {'cls': VWRSIPredictor, 'args': {"model_path": "vwrsi_multi_target_model.pth", "scaler_path": "vwrsi_seq_scaler.pkl"}}
    }
    
    initialized_count = 0
    for name, config in configs.items():
        predictor_class = config['cls']
        if predictor_class is not None: # Check if the class was successfully imported
            try:
                # Instantiate with appropriate arguments
                if name == 'adx_cci_macd' or name == 'candlestick':
                    instance = predictor_class(**config['args'])
                elif name == 'frama' or name == 'vwrsi':
                    instance = predictor_class(config['args']['model_path'], config['args']['scaler_path'])
                else:
                    instance = predictor_class()
                
                all_predictors[name] = instance
                logging.info(f"{name} Predictor initialized.")
                initialized_count += 1
            except Exception as e:
                logging.error(f"Failed to initialize {name} Predictor: {e}")
    
    if initialized_count == 0:
        logging.warning("No predictors were successfully initialized. Prediction will not function.")
    else:
        logging.info(f"Successfully initialized {initialized_count} predictor(s).")


# --- Helper Function to Get Candle Data ---
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
    Collects predictions for a given timeframe and returns the final trading decision
    along with raw predictions. Does NOT execute trades.
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
                predictions['boll'] = result.get('predictions', {})
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
                predictions['frama']['timeframe'] = timeframe_name
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
                predictions['vwrsi'] = vwrsi_result
                predictions['vwrsi']['timeframe'] = timeframe_name
            else:
                logging.error(f"Error in VW-RSI prediction for {timeframe_name}: {vwrsi_result['error']}")
                predictions['vwrsi'] = {"timeframe": timeframe_name, "error": f"Error: {vwrsi_result['error']}"}
        except Exception as e:
            logging.error(f"Error getting VW-RSI prediction for {timeframe_name}: {e}")
            predictions['vwrsi'] = {"timeframe": timeframe_name, "error": f"Error: {e}"}

    logging.info(f"--- {timeframe_name} Timeframe Predictions ---")
    if predictions:
        for predictor_name, result in predictions.items():
            logging.info(f"{predictor_name}: {result}")
    else:
        logging.info(f"No {timeframe_name} predictions available.")

    if predictions:
        final_dec = get_final_trading_signal(predictions)
        logging.info(f"--- {timeframe_name} Final Decision: {final_dec} ---")
        return {'decision': final_dec, 'timeframe': timeframe_name, 'symbol': symbol, 'raw_predictions': predictions}
    else:
        logging.info(f"Not enough {timeframe_name} predictions to make a final decision. Defaulting to Hold.")
        return {'decision': 'Hold', 'timeframe': timeframe_name, 'symbol': symbol, 'raw_predictions': {}}


# --- Main Logic to Run at Scheduled Intervals ---
def run_main_logic():
    """
    Contains the core prediction and trade execution logic that runs at scheduled times.
    """
    logging.info("Executing main trade logic at scheduled interval.")

    # Get decisions for both M30 and H1
    m30_data = get_timeframe_decision("M30", mt5.TIMEFRAME_M30, all_predictors)
    h1_data = get_timeframe_decision("H1", mt5.TIMEFRAME_H1, all_predictors)

    predictions_to_save = {
        "trade_magic_number": None,
        "M30_predictions": m30_data['raw_predictions'],
        "H1_predictions": h1_data['raw_predictions']
    }

    final_symbol = "XAUUSDm"
    final_decision_to_execute = "Hold"
    timeframe_decision_came_from = "N/A"

    if h1_data['decision'] in ["Buy", "Sell"]:
        final_decision_to_execute = h1_data['decision']
        timeframe_decision_came_from = "H1"
        print(f"Final Decision: {final_decision_to_execute} (from H1 - higher priority)")
        logging.info(f"H1 decision ({final_decision_to_execute}) takes precedence.")
    elif m30_data['decision'] in ["Buy", "Sell"]:
        final_decision_to_execute = m30_data['decision']
        timeframe_decision_came_from = "M30"
        print(f"Final Decision: {final_decision_to_execute} (from M30 - H1 was Hold)")
        logging.info(f"H1 decision is 'Hold', falling back to M30 decision ({final_decision_to_execute}).")
    else:
        final_decision_to_execute = "Hold"
        timeframe_decision_came_from = "N/A"
        print("Final Decision: Hold (Both H1 and M30 are Hold).")
        logging.info("Both H1 and M30 decisions are 'Hold'. No trade will be placed.")

    if execute_automated_trade and (final_decision_to_execute == "Buy" or final_decision_to_execute == "Sell"):
        current_positions = mt5.positions_get(symbol=final_symbol)

        if current_positions:
            print(f"Trade Info: Active positions found for {final_symbol}. Checking for conflicts.")
            logging.info(f"Active positions found for {final_symbol}. Checking for conflicts.")
            
            current_trade_type = "Buy" if current_positions[0].type == mt5.ORDER_TYPE_BUY else "Sell"
            
            if current_trade_type != final_decision_to_execute:
                print(f"Trade Info: Conflict detected. Skipping {final_decision_to_execute} trade for {final_symbol}.")
                logging.warning(f"Conflict: Current decision ({final_decision_to_execute}) from {timeframe_decision_came_from} differs from active trade ({current_trade_type}). Skipping trade.")
            else:
                print(f"Trade Info: Decision matches active trade. No new trade needed for {final_symbol}.")
                logging.info(f"Decision ({final_decision_to_execute}) from {timeframe_decision_came_from} matches active trade. No new trade needed.")
        else:
            print(f"Trade Info: Attempting to place {final_decision_to_execute} trade for {final_symbol} based on {timeframe_decision_came_from} decision.")
            logging.info(f"Attempting to place {final_decision_to_execute} trade for {final_symbol} based on {timeframe_decision_came_from} decision.")

            trade_magic_number = int(time.time())
            trade_success = False

            try:
                placer_result = execute_automated_trade(final_symbol, final_decision_to_execute)
                if isinstance(placer_result, tuple) and len(placer_result) == 2:
                    trade_success, trade_magic_number = placer_result
                else:
                    trade_success = placer_result
                    logging.warning("placer.py's execute_automated_trade did not return a magic number. Using timestamp as placeholder.")
            except Exception as e:
                logging.error(f"Error calling execute_automated_trade: {e}")
                print(f"Error: Failed to execute trade due to placer.py issue: {e}")

            if trade_success:
                predictions_to_save["trade_magic_number"] = trade_magic_number
                print(f"Trade Info: Trade successful for {final_symbol} - {final_decision_to_execute} from {timeframe_decision_came_from}. Magic Number: {trade_magic_number}")
                logging.info(f"Trade successful for {final_symbol} - {final_decision_to_execute} from {timeframe_decision_came_from}. Magic Number: {trade_magic_number}.")
                active_trades[final_symbol] = final_decision_to_execute

                json_filename = f"predictions_{trade_magic_number}.json"
                try:
                    with open(json_filename, 'w') as f:
                        json.dump(predictions_to_save, f, indent=4)
                    print(f"Trade Info: Raw predictions saved to {json_filename}")
                    logging.info(f"Raw predictions saved to {json_filename}")
                except Exception as e:
                    print(f"Error: Failed to save predictions to JSON: {e}")
                    logging.error(f"Error saving predictions to JSON: {e}")
            else:
                print(f"Trade Info: Trade failed for {final_symbol} - {final_decision_to_execute} from {timeframe_decision_came_from}.")
                logging.error(f"Trade failed for {final_symbol} - {final_decision_to_execute} from {timeframe_decision_came_from}.")
    elif not execute_automated_trade:
        print("Trade Info: placer.py or execute_automated_trade function not available. Automated trading skipped.")
        logging.warning("placer.py or execute_automated_trade function not available. Automated trading skipped.")
    else:
        print("Trade Info: Final combined decision is 'Hold'. No trade placed.")
        logging.info("Final combined decision is 'Hold'. No trade placed.")


# --- Main Execution Loop with Precise Timing ---
if __name__ == "__main__":
    if not mt5.initialize():
        print("Failed to initialize MetaTrader 5. Exiting.")
        logging.critical("Failed to initialize MetaTrader 5. Exiting.")
        exit()

    print("Starting main prediction process with 30-minute precise triggers...")
    logging.info("Starting main prediction process with 30-minute precise triggers...")

    # Initialize all predictors once at startup
    initialize_all_predictors_main()

    try:
        while True:
            now = datetime.now()
            current_minute = now.minute
            
            # Determine the target minutes for execution (00, 30)
            target_minutes = [0, 30]
            
            # Find the next closest target minute
            next_target_minute = None
            for tm in target_minutes:
                if tm >= current_minute: # Use >= to include current interval if it's the target
                    next_target_minute = tm
                    break
            
            # If current_minute is past the last target (e.g., 35 for a 00/30 schedule),
            # then the next target is the 00 minute of the next hour.
            if next_target_minute is None:
                next_target_minute = target_minutes[0] # This will be 0
                next_target_hour = (now.hour + 1) % 24
            else:
                next_target_hour = now.hour
            
            # Construct the next target datetime
            next_run_time = now.replace(
                hour=next_target_hour,
                minute=next_target_minute,
                second=0,
                microsecond=0
            )
            
            # If the calculated next_run_time is in the past (e.g., ran this minute already, or slightly behind)
            # advance it to the next valid interval.
            if next_run_time <= now:
                 # If it's already past current_minute, move to next 30min interval
                if current_minute >= next_target_minute: # e.g. now is 00:31, next target was 00:30
                    next_run_time += timedelta(minutes=30)
                    # Correct hour if timedelta rolls over
                    if next_run_time.minute < current_minute and next_run_time.hour == now.hour:
                        next_run_time += timedelta(hours=1)


            # Calculate sleep duration
            sleep_seconds = (next_run_time - now).total_seconds()
            
            # Ensure we don't sleep for negative time or too long if something went wrong
            if sleep_seconds < 0:
                sleep_seconds = 0 # Immediately re-evaluate
            
            print(f"Next run scheduled for {next_run_time.strftime('%H:%M:%S')}. Sleeping for {sleep_seconds:.1f} seconds.")
            logging.info(f"Next run scheduled for {next_run_time.strftime('%H:%M:%S')}. Sleeping for {sleep_seconds:.1f} seconds.")
            time.sleep(sleep_seconds)

            # After sleeping, check again if it's the correct interval to run
            now_after_sleep = datetime.now()
            current_interval_minute = (now_after_sleep.minute // 30) * 30 # Will be 0 or 30
            
            if current_interval_minute != last_main_run_minute_interval:
                # Execute the core main logic
                run_main_logic()
                last_main_run_minute_interval = current_interval_minute
            else:
                print(f"Skipping run at {now_after_sleep.strftime('%H:%M:%S')}. Already ran for this {current_interval_minute} minute interval.")
                logging.info(f"Skipping run at {now_after_sleep.strftime('%H:%M:%S')}. Already ran for this {current_interval_minute} minute interval.")

            time.sleep(1) # Small sleep to prevent busy-waiting immediately after execution

    except KeyboardInterrupt:
        print("Prediction process interrupted by user (Ctrl+C). Shutting down.")
        logging.info("Prediction process interrupted by user (Ctrl+C). Shutting down.")
    except Exception as e:
        print(f"A critical error occurred: {e}")
        logging.critical(f"A critical error occurred in the main prediction process: {e}", exc_info=True)
    finally:
        print("Shutting down MetaTrader5 and predictors.")
        logging.info("Shutting down MetaTrader5 and predictors.")
        try:
            mt5.shutdown()
        except Exception as e:
            print(f"Error during MT5 shutdown: {e}")
            logging.error(f"Error during MT5 shutdown: {e}")
