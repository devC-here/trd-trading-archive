import MetaTrader5 as mt5
import threading
import time
import logging
import json
from datetime import datetime, timedelta

# Predictor/Logic imports - set to None if import fails
try:
    from adxmacdccideploy import MarketPredictor
    from bollingerdeploy import TradingPredictor as BollingerTradingPredictor
    from candledeploy import CandleSequencePredictor
    from FRAMAPredictor import FRAMAPredictor
    from positiondeploy import BollingerPredictor as PositionBollingerPredictor
    from vwrsideploy import VWRSIPredictor
    from fusion_engine import get_final_trading_signal
except ImportError as e:
    logging.error(f"Predictor import error: {e}. Features limited.")
    MarketPredictor, BollingerTradingPredictor, CandleSequencePredictor, FRAMAPredictor, PositionBollingerPredictor, VWRSIPredictor, get_final_trading_signal = [None] * 7

try:
    from countelogic import compare
except ImportError as e:
    logging.error(f"Error importing 'compare': {e}. Close decision unavailable.")
    compare = None

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=[logging.FileHandler("prediction_log.log")])

# Global Variables
PRIMARY_SYMBOL = "XAUUSDm"
all_predictors = {}
ongoing_trade_magic_numbers = []
all_timeframe_predictions_data = {}
last_prediction_run_minute_interval = -1

# Initialize All Predictors
def initialize_all_predictors():
    global all_predictors
    configs = {
        'adx_cci_macd': {'cls': MarketPredictor, 'mth': 'get_prediction', 'proc': lambda r, tf: r, 'args': ([PRIMARY_SYMBOL],)},
        'boll': {'cls': BollingerTradingPredictor, 'mth': 'predict_timeframe', 'proc': lambda r, tf: r.get('predictions', {}) if "error" not in r else r, 'args': ()},
        'candlestick': {'cls': CandleSequencePredictor, 'mth': 'get_most_confident_trade', 'proc': lambda r, tf: r, 'args': ([PRIMARY_SYMBOL],)},
        'frama': {'cls': FRAMAPredictor, 'mth': 'predict', 'proc': lambda r, tf: (r['predictions'].update({'timeframe': tf}) or r['predictions']) if "error" not in r else r, 'args': ("frama_multi_target_model.pth", "frama_seq_scaler.pkl")},
        'position_bollinger': {'cls': PositionBollingerPredictor, 'mth': 'predict', 'proc': lambda r, tf: r if "error" not in r else r, 'args': ()},
        'vwrsi': {'cls': VWRSIPredictor, 'mth': 'predict', 'proc': lambda r, tf: (r.update({'timeframe': tf}) or r) if "error" not in r else r, 'args': ("vwrsi_multi_target_model.pth", "vwrsi_seq_scaler.pkl")}
    }
    for name, cfg in configs.items():
        if cfg['cls']:
            try:
                all_predictors[name] = {'inst': cfg['cls'](*cfg.get('args', ())), 'mth': cfg['mth'], 'proc': cfg['proc']}
                logging.info(f"{name} initialized.")
            except Exception as e:
                logging.error(f"Failed {name} init: {e}")
    if not all_predictors: logging.warning("No predictors initialized.")

# Call Predictor Method
def _call_predictor(name, info, sym, tf_name, tf_code):
    try:
        method = getattr(info['inst'], info['mth'])
        res = method(sym, tf_code) if info['mth'] == 'get_prediction' else \
              method(tf_name, tf_code) if info['mth'] == 'predict_timeframe' else \
              method(tf_code)
        return info['proc'](res, tf_name)
    except Exception as e: return {"error": f"Error {name}: {e}"}

# Collect Predictions
def get_predictions(sym, tf_name, tf_code, preds):
    results = {}
    for name, info in preds.items():
        if info['inst'] is not None: results[name] = _call_predictor(name, info, sym, tf_name, tf_code)
    
    final_sig = "Hold"
    if get_final_trading_signal and results:
        try: final_sig = get_final_trading_signal(results)
        except Exception as e: logging.error(f"Fusion engine error: {e}")
    else: logging.warning("Cannot get final signal.")
    
    logging.info(f"Predictions for {sym}-{tf_name}: {json.dumps(results)}")
    logging.info(f"Final Signal {sym}-{tf_name}: {final_sig}")
    return {'symbol': sym, 'timeframe': tf_name, 'raw_predictions': results, 'final_signal': final_sig}

# Get Ongoing Trade Magic Numbers
def get_ongoing_magic_numbers_at_interval():
    global ongoing_trade_magic_numbers
    try:
        positions = mt5.positions_get()
        if positions:
            current_magic_numbers = [pos.ticket for pos in positions]
            if current_magic_numbers != ongoing_trade_magic_numbers:
                ongoing_trade_magic_numbers = current_magic_numbers
                logging.info(f"Ongoing trade magic numbers updated: {ongoing_trade_magic_numbers}")
        else:
            if ongoing_trade_magic_numbers:
                ongoing_trade_magic_numbers = []
                logging.info("No ongoing trades found. Magic number list cleared.")
    except Exception as e:
        logging.error(f"Error fetching ongoing trade magic numbers: {e}")

# Close a Trade by Magic Number (Position Ticket)
def close_trade(magic_number):
    position = mt5.positions_get(ticket=magic_number)
    if not position: logging.warning(f"No position {magic_number} found to close."); return False
    pos_info = position[0]
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": pos_info.symbol,
        "volume": pos_info.volume,
        "type": mt5.ORDER_TYPE_SELL if pos_info.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "position": pos_info.ticket,
        "price": mt5.symbol_info_tick(pos_info.symbol).bid if pos_info.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(pos_info.symbol).ask,
        "deviation": 20,
        "magic": pos_info.magic,
        "comment": "Closed by Observer",
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE: logging.info(f"✅ Trade {magic_number} closed. Res: {result.comment}"); return True
    else: logging.error(f"❌ Failed to close {magic_number}. Err: {result.comment} ({result.retcode})"); return False

# Prediction and Data Monitoring Thread
def monitor_predictions_and_data_thread():
    global all_timeframe_predictions_data, last_prediction_run_minute_interval, ongoing_trade_magic_numbers
    logging.info("Starting data monitoring with 15-minute precise triggers.")
    timeframes_to_monitor = {"M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1}
    
    while True:
        now = datetime.now()
        current_minute = now.minute
        target_minutes = [0, 15, 30, 45]
        
        next_target_minute = next((tm for tm in target_minutes if tm > current_minute), target_minutes[0])
        next_target_hour = now.hour if next_target_minute > current_minute else (now.hour + 1) % 24
        
        next_run_time = now.replace(hour=next_target_hour, minute=next_target_minute, second=0, microsecond=0)
        if next_run_time <= now: next_run_time += timedelta(minutes=15)
        
        sleep_seconds = (next_run_time - now).total_seconds()
        if sleep_seconds < 0: sleep_seconds = 0
        
        logging.info(f"Next update scheduled for {next_run_time.strftime('%H:%M:%S')}. Sleeping {sleep_seconds:.1f}s.")
        time.sleep(sleep_seconds)

        now_after_sleep = datetime.now()
        current_interval_minute = (now_after_sleep.minute // 15) * 15
        
        if current_interval_minute != last_prediction_run_minute_interval:
            get_ongoing_magic_numbers_at_interval() # Update magic numbers

            if not all_predictors: logging.warning("No predictors. Skipping prediction gathering.")
            else:
                for tf_name, tf_code in timeframes_to_monitor.items():
                    all_timeframe_predictions_data[tf_name] = get_predictions(PRIMARY_SYMBOL, tf_name, tf_code, all_predictors)
                logging.info(f"Data updated for {PRIMARY_SYMBOL} at {now_after_sleep.strftime('%H:%M:%S')} (Interval {current_interval_minute}).")

                magic_numbers_to_close = []
                if compare:
                    try: magic_numbers_to_close = compare(ongoing_trade_magic_numbers, all_timeframe_predictions_data)
                    except Exception as e: logging.error(f"Error calling 'compare': {e}")
                else: logging.warning("'compare' not available. Skipping close decision.")
            
            if magic_numbers_to_close:
                logging.info(f"Attempting to close trades: {magic_numbers_to_close}")
                for mn in magic_numbers_to_close: close_trade(mn)
            else: logging.info("No trades to close identified.")

            last_prediction_run_minute_interval = current_interval_minute
        else:
            logging.info(f"Skipping update at {now_after_sleep.strftime('%H:%M:%S')}. Already updated for {current_interval_minute}.")

        time.sleep(1) # Small sleep to prevent busy-waiting


if __name__ == "__main__":
    if not mt5.initialize(): logging.critical("MT5 init failed. Exiting."); exit()
    logging.info("System starting.")
    initialize_all_predictors()
    
    main_monitor_thread = threading.Thread(target=monitor_predictions_and_data_thread, daemon=True)
    main_monitor_thread.start()

    try: while True: time.sleep(10)
    except KeyboardInterrupt: logging.info("Process interrupted.")
    except Exception as e: logging.critical(f"Critical error: {e}", exc_info=True)
    finally:
        logging.info("Shutting down.")
        try: mt5.shutdown()
        except Exception as e: logging.error(f"MT5 shutdown error: {e}")