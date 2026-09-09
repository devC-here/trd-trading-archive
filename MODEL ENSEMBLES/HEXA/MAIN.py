import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import torch
import joblib
import warnings
import logging
import time
import json
from datetime import datetime, timedelta

# Suppress all warnings from being printed to the console
warnings.filterwarnings("ignore")

# Import all predictor classes from the individual files
from adxmacdccideploy import MarketPredictor
from bollingerdeploy import TradingPredictor as BollingerTradingPredictor
from candledeploy import CandleSequencePredictor
from FRAMAdeploy import FRAMAPredictor
from positiondeploy import BollingerPredictor as PositionBollingerPredictor
from vwrsideploy import VWRSIPredictor

# Re-added: Import the final decision function from fusion_engine
from fusion_engine import get_final_trading_signal

# Removed: Import the elaboration function and Gemini AI function
# try:
#     from elaboratinglogic import elaborate_market_conditions
# except ImportError:
#     print("Error: elaboratinglogic.py not found. AI-driven decisions will be disabled.")
#     elaborate_market_conditions = None

# try:
#     from ai import get_gemini_trade_advice
# except ImportError:
#     print("Error: ai.py not found or get_gemini_trade_advice not available. AI-driven decisions will be disabled.")
#     get_gemini_trade_advice = None

# Import trade placing function (assuming placer.py exists)
try:
    from placer import execute_automated_trade
except ImportError:
    print("Error: placer.py not found. Automated trading will be disabled.")
    execute_automated_trade = None

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("trading_log.log")
    ]
)

# --- Global Variables for Trade Management and Timing ---
last_main_run_minute_interval = -1

# Global dictionary to hold initialized predictor instances
all_predictors = {}

# --- JSON file for storing all trade data ---
ALL_TRADING_RECORDS_FILE = "all_trading_records.json"

# --- Helper Function to Initialize All Predictors ---
def initialize_all_predictors_main():
    """
    Initializes all market predictor models.
    Stores initialized predictor instances in the global 'all_predictors' dictionary.
    """
    global all_predictors
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
        if predictor_class is not None:
            try:
                if name in ['adx_cci_macd', 'candlestick']:
                    instance = predictor_class(**config['args'])
                elif name in ['frama', 'vwrsi']:
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


# --- Helper Function to Get Candle Data (kept for completeness, not used in this refactor's core logic) ---
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
    symbol = "XAUUSDm"

    if not predictors:
        logging.error(f"Predictors not initialized for {timeframe_name}. Skipping prediction.")
        return {'decision': 'Hold', 'timeframe': timeframe_name, 'symbol': symbol, 'raw_predictions': {"error": "Predictors not initialized"}}

    for predictor_name, predictor_instance in predictors.items():
        try:
            if predictor_name == 'adx_cci_macd':
                predictions['adx_cci_macd'] = predictor_instance.get_prediction(symbol, timeframe_code)
            elif predictor_name == 'boll': # This is the BollingerTradingPredictor (advanced)
                result = predictor_instance.predict_timeframe(timeframe_name, timeframe_code)
                predictions['boll'] = result.get('predictions', {}) if "error" not in result else {"error": result['error']}
            elif predictor_name == 'candlestick':
                predictions['candlestick'] = predictor_instance.get_most_confident_trade(timeframe_code)
            elif predictor_name == 'frama':
                frama_result = predictor_instance.predict(timeframe_code)
                predictions['frama'] = frama_result['predictions'] if "error" not in frama_result else {"error": frama_result['error']}
                if 'frama' in predictions and "error" not in predictions['frama']:
                    predictions['frama']['timeframe'] = timeframe_name
            elif predictor_name == 'position_bollinger': # This is the BollingerPredictor (raw/rule-based)
                result = predictor_instance.predict(timeframe_code)
                predictions['position_bollinger'] = result.get('predictions', {}) if "error" not in result else {"error": result['error']}
            elif predictor_name == 'vwrsi':
                vwrsi_result = predictor_instance.predict(timeframe_code)
                predictions['vwrsi'] = vwrsi_result if "error" not in vwrsi_result else {"error": vwrsi_result['error']}
                if 'vwrsi' in predictions and "error" not in predictions['vwrsi']:
                    predictions['vwrsi']['timeframe'] = timeframe_name
        except Exception as e:
            logging.error(f"Error getting {predictor_name} prediction for {timeframe_name}: {e}")
            predictions[predictor_name] = {"error": f"Error: {e}"}

    logging.info(f"--- {timeframe_name} Timeframe Predictions ---")
    if predictions:
        for predictor_name, result in predictions.items():
            logging.info(f"{predictor_name}: {result}")
    else:
        logging.info(f"No {timeframe_name} predictions available.")

    # Call the robust get_final_trading_signal from fusion_engine
    final_dec = get_final_trading_signal(predictions)
    logging.info(f"--- {timeframe_name} Final Decision: {final_dec} ---")
    return {'decision': final_dec, 'timeframe': timeframe_name, 'symbol': symbol, 'raw_predictions': predictions}

# --- Helper function for saving to JSON (Moved to top-level) ---
def _save_trading_record(record_to_add, current_records_dict):
    """
    Helper to save a single record to the JSON file, ensuring the file structure is maintained.
    This handles loading, appending/updating, and saving.
    
    Args:
        record_to_add (dict): The dictionary representing the current prediction/trade record.
        current_records_dict (dict): The dictionary containing "trades" and "prediction_logs"
                                     that was loaded or initialized. This is passed to avoid re-loading.
    """
    # For safety, ensure the trades dict and prediction_logs list exist
    if "trades" not in current_records_dict:
        current_records_dict["trades"] = {}
    if "prediction_logs" not in current_records_dict:
        current_records_dict["prediction_logs"] = []

    current_records_dict["prediction_logs"].append(record_to_add)

    # If the record_to_add itself contains a magic number indicating a placed trade,
    # add it to the 'trades' dictionary as well.
    if record_to_add.get("trade_magic_number_if_placed") is not None:
        magic_key = record_to_add["trade_magic_number_if_placed"]
        current_records_dict["trades"][magic_key] = {
            "timestamp": record_to_add["timestamp"],
            "trade_direction": record_to_add["trade_direction_if_placed"],
            "decision_timeframe_source": record_to_add["decision_timeframe_source"],
            "mt5_ticket": record_to_add["mt5_ticket_if_placed"],
            "M15_predictions_at_trade_time": record_to_add["M15_predictions"],
            "M30_predictions_at_trade_time": record_to_add["M30_predictions"],
            "H1_predictions_at_trade_time": record_to_add["H1_predictions"]
        }

    try:
        with open(ALL_TRADING_RECORDS_FILE, 'w') as f:
            json.dump(current_records_dict, f, indent=4)
        logging.info(f"All trading records updated and saved to {ALL_TRADING_RECORDS_FILE}")
    except Exception as e:
        logging.error(f"Error saving all trading records to JSON: {e}")


# --- Main Logic to Run at Scheduled Intervals ---
def run_main_logic(current_run_time):
    """
    Contains the core prediction and trade execution logic that runs at scheduled times.
    Accepts current_run_time to check the minute for conditional trade placement.
    """
    logging.info(f"Executing main trade logic at scheduled interval: {current_run_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize all timeframe data to default 'Hold' and empty raw predictions
    m15_data = {'decision': 'Hold', 'timeframe': 'M15', 'symbol': 'XAUUSDm', 'raw_predictions': {}}
    m30_data = {'decision': 'Hold', 'timeframe': 'M30', 'symbol': 'XAUUSDm', 'raw_predictions': {}}
    h1_data = {'decision': 'Hold', 'timeframe': 'H1', 'symbol': 'XAUUSDm', 'raw_predictions': {}}

    final_symbol = "XAUUSDm"
    final_decision_to_execute = "Hold" # Default decision
    timeframe_decision_came_from = "N/A" # Default, will be M30, H1, or M15 based on time
    
    trade_magic_number = int(time.time()) # Default magic number
    trade_direction = "N/A" # Default direction

    current_minute = current_run_time.minute

    # --- Apply time-based priority logic for fetching predictions and making decision ---
    if current_minute == 0: # At XX:00:00, calculate M30 and use its decision
        m30_data = get_timeframe_decision("M30", mt5.TIMEFRAME_M30, all_predictors)
        if m30_data['decision'] in ["Buy", "Sell"]:
            final_decision_to_execute = m30_data['decision']
            timeframe_decision_came_from = "M30"
            logging.info(f"Final Decision: {final_decision_to_execute} (from M30 - 00-minute priority)")
            print(f"Final Decision: {final_decision_to_execute} (from M30 - 00-minute priority)")
        else:
            final_decision_to_execute = "Hold"
            timeframe_decision_came_from = "M30 (No Signal)"
            logging.info("Final Decision: Hold (M30 at 00-minute mark returned no signal).")
            print("Final Decision: Hold (M30 at 00-minute mark returned no signal).")
    elif current_minute == 30: # At XX:30:00, calculate H1 and use its decision
        h1_data = get_timeframe_decision("H1", mt5.TIMEFRAME_H1, all_predictors)
        if h1_data['decision'] in ["Buy", "Sell"]:
            final_decision_to_execute = h1_data['decision']
            timeframe_decision_came_from = "H1"
            logging.info(f"Final Decision: {final_decision_to_execute} (from H1 - 30-minute priority)")
            print(f"Final Decision: {final_decision_to_execute} (from H1 - 30-minute priority)")
        else:
            final_decision_to_execute = "Hold"
            timeframe_decision_came_from = "H1 (No Signal)"
            logging.info("Final Decision: Hold (H1 at 30-minute mark returned no signal).")
            print("Final Decision: Hold (H1 at 30-minute mark returned no signal).")
    elif current_minute in [15, 45]: # At XX:15:00 or XX:45:00, calculate M15 and use its decision
        m15_data = get_timeframe_decision("M15", mt5.TIMEFRAME_M15, all_predictors)
        if m15_data['decision'] in ["Buy", "Sell"]:
            final_decision_to_execute = m15_data['decision']
            timeframe_decision_came_from = "M15"
            logging.info(f"Final Decision: {final_decision_to_execute} (from M15 - {current_minute}-minute priority)")
            print(f"Final Decision: {final_decision_to_execute} (from M15 - {current_minute}-minute priority)")
        else:
            final_decision_to_execute = "Hold"
            timeframe_decision_came_from = f"M15 (No Signal)"
            logging.info(f"Final Decision: Hold (M15 at {current_minute}-minute mark returned no signal).")
            print(f"Final Decision: Hold (M15 at {current_minute}-minute mark returned no signal).")
    else: # Should not happen with the main loop's scheduling, but as a fallback
        final_decision_to_execute = "Hold"
        timeframe_decision_came_from = "N/A (Unexpected Minute)"
        logging.info(f"Final Decision: Hold (Unexpected minute {current_minute}).")
        print(f"Final Decision: Hold (Unexpected minute {current_minute}).")


    if final_decision_to_execute in ["Buy", "Sell"]:
        trade_direction = final_decision_to_execute

    # Extract raw predictions for saving (from potentially updated data)
    raw_m15_preds = m15_data['raw_predictions']
    raw_m30_preds = m30_data['raw_predictions']
    raw_h1_preds = h1_data['raw_predictions']

    # Initialize a record for the current prediction run based on Fusion Engine decision
    current_prediction_record = {
        "timestamp": current_run_time.strftime("%Y-%m-%d %H:%M:%S"),
        "M15_predictions": raw_m15_preds,
        "M30_predictions": raw_m30_preds,
        "H1_predictions": raw_h1_preds,
        "overall_decision": final_decision_to_execute,
        "decision_timeframe_source": timeframe_decision_came_from,
        "trade_magic_number_if_placed": None,
        "trade_direction_if_placed": None,
        "mt5_ticket_if_placed": None
    }

    # Load existing data or initialize the structure for saving
    all_trading_records = {"trades": {}, "prediction_logs": []}
    try:
        with open(ALL_TRADING_RECORDS_FILE, 'r') as f:
            loaded_data = json.load(f)
            all_trading_records["trades"] = loaded_data.get("trades", {})
            all_trading_records["prediction_logs"] = loaded_data.get("prediction_logs", [])
    except FileNotFoundError:
        logging.info(f"'{ALL_TRADING_RECORDS_FILE}' not found. Creating a new one.")
    except json.JSONDecodeError:
        logging.warning(f"Error decoding '{ALL_TRADING_RECORDS_FILE}'. Starting with empty data.")
        all_trading_records = {"trades": {}, "prediction_logs": []}


    # --- Conditional Trade Execution Logic ---
    # This block remains the same, as the decision is already made above
    if current_minute in [0, 30]: # Trades only at 00 and 30 minute marks
        if execute_automated_trade and (final_decision_to_execute == "Buy" or final_decision_to_execute == "Sell"):
            current_positions = mt5.positions_get(symbol=final_symbol)

            if current_positions:
                logging.info(f"Active positions found for {final_symbol}. Checking for conflicts.")
                current_trade_type = "Buy" if current_positions[0].type == mt5.ORDER_TYPE_BUY else "Sell"

                if current_trade_type != final_decision_to_execute:
                    logging.warning(f"Conflict: Current decision ({final_decision_to_execute}) differs from active trade ({current_trade_type}). Skipping trade.")
                    print(f"Conflict: Current decision ({final_decision_to_execute}) differs from active trade ({current_trade_type}). Skipping trade.")
                else:
                    logging.info(f"Decision ({final_decision_to_execute}) matches active trade. No new trade needed.")
                    print(f"Decision ({final_decision_to_execute}) matches active trade. No new trade needed.")
            else:
                logging.info(f"Attempting to place {final_decision_to_execute} trade for {final_symbol} based on Fusion Engine decision.")
                print(f"Attempting to place {final_decision_to_execute} trade for {final_symbol} based on Fusion Engine decision.")

                placer_trade_magic = trade_magic_number # Default to timestamp magic number
                mt5_trade_ticket = None # Initialize MT5 ticket
                trade_success = False

                try:
                    placer_result = execute_automated_trade(final_symbol, final_decision_to_execute)
                    
                    if isinstance(placer_result, tuple):
                        if len(placer_result) == 3: # If placer returns success, magic, and ticket
                            trade_success, placer_trade_magic, mt5_trade_ticket = placer_result
                        elif len(placer_result) == 2: # If placer returns success and magic (older version)
                            trade_success, placer_trade_magic = placer_result
                            logging.warning("placer.py's execute_automated_trade did not return ticket number. Ticket will be 'None'.")
                        else:
                            trade_success = False
                            logging.error(f"placer.py returned unexpected tuple length: {len(placer_result)}")
                    else: # If placer returns just success bool
                        trade_success = placer_result
                        logging.warning("placer.py's execute_automated_trade did not return magic or ticket number. Using timestamp as placeholder for magic.")
                except Exception as e:
                    logging.error(f"Error calling execute_automated_trade: {e}")

                if trade_success:
                    current_prediction_record["trade_magic_number_if_placed"] = str(placer_trade_magic)
                    current_prediction_record["trade_direction_if_placed"] = final_decision_to_execute
                    current_prediction_record["mt5_ticket_if_placed"] = mt5_trade_ticket

                    # The _save_trading_record function will handle adding this to 'trades'
                    print(f"Trade successfully placed: {final_decision_to_execute} {final_symbol}. Magic: {placer_trade_magic}, Ticket: {mt5_trade_ticket}.") # Print success
                    logging.info(f"Trade successful for {final_symbol} - {final_decision_to_execute} from {timeframe_decision_came_from}. Magic Number: {placer_trade_magic}, Ticket: {mt5_trade_ticket}.")

                else:
                    print(f"Trade failed for {final_symbol} - {final_decision_to_execute}.") # Print failure
                    logging.error(f"Trade failed for {final_symbol} - {final_decision_to_execute} from {timeframe_decision_came_from}.")
        elif not execute_automated_trade:
            print("Automated trading is disabled (placer.py not found). Skipping trade placement.") # Print this
            logging.warning("placer.py or execute_automated_trade function not available. Automated trading skipped.")
        else:
            print("Final combined Fusion Engine decision is 'Hold'. No trade placed at this 00/30 minute mark.") # Print specific hold reason
            logging.info("Final combined decision is 'Hold'. No trade placed.")
    else:
        # This else block is for 15 and 45 minute marks where trade execution is skipped
        print(f"Current minute is {current_minute}. Skipping trade placement (not a 00 or 30 minute mark), but predictions are still saved.") # Print this
        logging.info(f"Current minute is {current_minute}. Skipping trade placement, but predictions are still saved.")

    # --- Save all data to the JSON file ---
    # Call the helper function to save all records
    _save_trading_record(current_prediction_record, all_trading_records)


# --- Entry Point ---
if __name__ == "__main__":
    if not mt5.initialize():
        print("Failed to initialize MetaTrader 5. Exiting.")
        logging.critical("Failed to initialize MetaTrader 5. Exiting.")
        exit()

    print("Starting main prediction process with 15-minute precise triggers for predictions, and time-based Fusion Engine-driven trades...")
    logging.info("Starting main prediction process with 15-minute precise triggers for predictions, and time-based Fusion Engine-driven trades...")

    initialize_all_predictors_main()

    try:
        while True:
            now = datetime.now()
            
            # Calculate the next target time (00, 15, 30, 45) at second 0
            # This ensures we always jump to the next *future* valid 15-minute mark
            target_minutes = [0, 15, 30, 45]
            
            # Start with the current time, and advance its minute to the next valid 15-minute interval
            # Ensure next_run_time is always in the future
            next_run_time = now.replace(second=0, microsecond=0) # Reset seconds and microseconds
            
            # If current time is not precisely on target, find the next one
            while next_run_time < now or next_run_time.minute not in target_minutes:
                next_run_time += timedelta(minutes=1) # Increment by minute
                # Adjust to the nearest 15-minute mark if we crossed one
                if next_run_time.minute % 15 == 0 and next_run_time > now:
                    break # Found the next target minute
                elif next_run_time.minute % 15 != 0 and next_run_time.minute < now.minute and next_run_time.hour == now.hour:
                    # If we've passed a 15-min mark in the current hour and not landed exactly on it
                    # This handles cases like 13:46 -> next is 14:00 (not 13:45)
                    next_run_time = next_run_time.replace(minute=(next_run_time.minute // 15 + 1) * 15)
                    if next_run_time.minute == 60: # Handle hour rollover
                        next_run_time = next_run_time.replace(minute=0, hour=(next_run_time.hour + 1) % 24)
                    next_run_time = next_run_time.replace(second=0, microsecond=0)
                    if next_run_time >= now: # Ensure it's not in the past
                        break
            
            sleep_seconds = (next_run_time - now).total_seconds()
            
            if sleep_seconds < 0: # Failsafe, should not happen with robust calculation
                sleep_seconds = 0
            
            print(f"Next run scheduled for {next_run_time.strftime('%H:%M:%S')}. Sleeping for {sleep_seconds:.1f} seconds.")
            logging.info(f"Next run scheduled for {next_run_time.strftime('%H:%M:%S')}. Sleeping for {sleep_seconds:.1f} seconds.")
            time.sleep(sleep_seconds)

            # After sleeping, verify the current time is exactly on the target interval
            now_after_sleep = datetime.now()
            # Run if it's on a target minute and hasn't run for this specific minute interval yet.
            if (now_after_sleep.minute in target_minutes) and \
               (last_main_run_minute_interval != now_after_sleep.minute):

                run_main_logic(now_after_sleep)
                last_main_run_minute_interval = now_after_sleep.minute
            else:
                print(f"Skipping run at {now_after_sleep.strftime('%H:%M:%S')}. Not exactly on interval or already ran.")
                logging.info(f"Skipping run at {now_after_sleep.strftime('%H:%M:%S')}. Not exactly on interval or already ran.")

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