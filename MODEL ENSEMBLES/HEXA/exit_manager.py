import MetaTrader5 as mt5
import json
import os
from datetime import datetime, timedelta
import time
import re # Import regex for parsing AI advice

# Import the elaboration function (ensure elaboratinglogic.py is in the same directory)
try:
    from elaboratinglogic import elaborate_market_conditions
except ImportError:
    print("Error: elaboratinglogic.py not found. Cannot elaborate market conditions.")
    elaborate_market_conditions = None

# Import the Gemini interaction function (ensure ai.py is in the same directory)
try:
    from ai import get_gemini_trade_advice
except ImportError:
    print("Error: ai.py not found or get_gemini_trade_advice not available. AI advice will be disabled.")
    get_gemini_trade_advice = None

# --- Configuration ---
ALL_TRADING_RECORDS_FILE = "all_trading_records.json"

def load_trading_records():
    """
    Loads all trading records from the JSON file.
    Returns a dictionary with 'trades' and 'prediction_logs' or an empty structure if file not found/invalid.
    """
    if not os.path.exists(ALL_TRADING_RECORDS_FILE):
        print(f"'{ALL_TRADING_RECORDS_FILE}' not found. No records to display.")
        return {"trades": {}, "prediction_logs": []}

    try:
        with open(ALL_TRADING_RECORDS_FILE, 'r') as f:
            data = json.load(f)
            data["trades"] = data.get("trades", {})
            data["prediction_logs"] = data.get("prediction_logs", [])
            return data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{ALL_TRADING_RECORDS_FILE}': {e}")
        print("The file might be corrupted or empty. Returning empty records.")
        return {"trades": {}, "prediction_logs": []}
    except Exception as e:
        print(f"An unexpected error occurred while loading records: {e}")
        return {"trades": {}, "prediction_logs": []}

def get_latest_prediction_data_for_elaboration(records):
    """
    Retrieves and returns the raw prediction data of the very latest log entry,
    formatted for input to elaborate_market_conditions.
    """
    if records["prediction_logs"]:
        latest_log = records["prediction_logs"][-1]
        return {
            "M15_predictions": latest_log.get("M15_predictions", {}),
            "M30_predictions": latest_log.get("M30_predictions", {}),
            "H1_predictions": latest_log.get("H1_predictions", {})
        }
    return None

def close_position(position_ticket, symbol, volume, trade_type):
    """
    Closes an open MetaTrader 5 position.
    Args:
        position_ticket (int): The ticket number of the position to close.
        symbol (str): The symbol of the position (e.g., "XAUUSDm").
        volume (float): The volume of the position (lot size).
        trade_type (int): The type of the position (mt5.ORDER_TYPE_BUY or mt5.ORDER_TYPE_SELL).
    Returns:
        bool: True if the position was successfully closed, False otherwise.
    """
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_SELL if trade_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY, # Opposite of open trade
        "position": position_ticket,
        "price": mt5.symbol_info_tick(symbol).bid if trade_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask,
        "deviation": 20, # Slippage in points
        "magic": position_ticket, # Using ticket as magic for closure
        "comment": "AI-advised close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    result = mt5.OrderSend(request)

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"--- SUCCESS: Position {position_ticket} ({symbol} {trade_type}) closed. ---")
        return True
    else:
        print(f"--- FAILED to close position {position_ticket} ({symbol} {trade_type}). ---")
        print(f"   Return code: {result.retcode}")
        print(f"   Description: {mt5.last_error()}")
        return False


def analyze_trades_with_ai_guidance():
    """
    Connects to MT5, retrieves open trades, finds their associated prediction data,
    gets the latest overall prediction, elaborates on both, and seeks AI advice
    based on the comparison. Includes functionality to close trades based on AI advice.
    """
    print("\n--- Starting AI-Powered Trade and Prediction Analysis Cycle ---")

    # MT5 is initialized once in the __main__ block. We assume it's active here.

    # 2. Load all trading records
    records = load_trading_records()
    if not records["trades"] and not records["prediction_logs"]:
        print("No trading records found or loaded. Skipping detailed analysis for this cycle.")
        return

    # Get the latest market conditions once for comparison with all open trades
    print("\n--- Retrieving Latest Market Conditions ---")
    latest_raw_prediction_data = get_latest_prediction_data_for_elaboration(records)
    if not latest_raw_prediction_data:
        print("No latest prediction logs available. Cannot perform comparative analysis for this cycle.")
        return

    elaborated_latest_conditions = None
    if elaborate_market_conditions:
        elaborated_latest_conditions = elaborate_market_conditions(latest_raw_prediction_data)
        print("Latest Market Conditions Elaborated.")
    else:
        print("Elaboration function (elaborate_market_conditions) not available. Skipping AI analysis.")
        return

    # 3. Retrieve Open Trades from MT5 and get AI advice for them
    print("\n--- Analyzing Open Trades and Seeking AI Advice ---")
    positions = mt5.positions_get()

    if positions:
        print(f"Found {len(positions)} open trade(s).")
        for position in positions:
            magic_number = str(position.magic)
            trade_type_str = "BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"
            symbol = position.symbol
            ticket = position.ticket
            volume = position.volume # Get the volume (lot size)

            print(f"\n--- Open Trade: Ticket {ticket}, Symbol: {symbol}, Type: {trade_type_str}, Magic: {magic_number}, Volume: {volume} ---")

            trade_record = records["trades"].get(magic_number)
            if trade_record:
                trade_predictions_at_entry = {
                    "M15_predictions": trade_record.get("M15_predictions_at_trade_time", {}),
                    "M30_predictions": trade_record.get("M30_predictions_at_trade_time", {}),
                    "H1_predictions": trade_record.get("H1_predictions_at_trade_time", {})
                }

                if elaborate_market_conditions:
                    print("Elaborating Market Conditions at Trade Entry...")
                    elaborated_trade_entry_conditions = elaborate_market_conditions(trade_predictions_at_entry)

                    if get_gemini_trade_advice:
                        print("\nGetting AI Advice for this Open Trade (Comparative Analysis)...")
                        ai_prompt = (
                            f"Act as a seasoned forex trading advisor. You are reviewing an open trade.\n"
                            f"The trade is a '{trade_type_str}' on '{symbol}' with Magic Number '{magic_number}'.\n\n"
                            f"--- Market Conditions at Trade Entry (Timestamp: {trade_record.get('timestamp', 'N/A')}) ---\n"
                            f"{json.dumps(elaborated_trade_entry_conditions, indent=4)}\n\n"
                            f"--- Latest Market Conditions (Timestamp: {records['prediction_logs'][-1].get('timestamp', 'N/A')}) ---\n"
                            f"{json.dumps(elaborated_latest_conditions, indent=4)}\n\n"
                            f"Your task is to compare the market conditions at the time the trade was opened with the latest market conditions. "
                            f"Considering the original trade direction ('{trade_type_str}'), advise whether to:\n"
                            f"1. **CONTINUE HOLDING**: Market conditions still support the original trade direction or have become more favorable.\n"
                            f"2. **CLOSE TRADE**: Market conditions have significantly deteriorated or reversed against the original trade direction.\n"
                            f"3. **CONSIDER HEDGING**: Market conditions show uncertainty, strong opposing signals, or increased volatility that warrants risk mitigation.\n\n"
                            f"Provide a concise recommendation (CONTINUE HOLDING / CLOSE TRADE / CONSIDER HEDGING) and a brief (1-3 sentences) justification for your decision based on the changes observed between the two time points."
                        )
                        ai_advice = get_gemini_trade_advice(ai_prompt)
                        print(f"AI Advisor's Recommendation for Trade {ticket}: {ai_advice}")

                        # --- Automated Trade Closure Logic ---
                        if "CLOSE TRADE" in ai_advice.upper():
                            print(f"\nAI recommended to CLOSE TRADE for position {ticket}. Attempting to close...")
                            # Determine MT5 trade type for closure based on original trade type
                            mt5_trade_type_for_closure = position.type
                            
                            # Fetch latest symbol info for price
                            symbol_info = mt5.symbol_info(symbol)
                            if symbol_info is None:
                                print(f"Error: Could not get symbol info for {symbol}. Cannot close trade.")
                                continue # Skip to next position

                            tick_info = mt5.symbol_info_tick(symbol)
                            if tick_info is None:
                                print(f"Error: Could not get tick info for {symbol}. Cannot close trade.")
                                continue # Skip to next position

                            # The price for closing depends on the type of the original position
                            # If original was BUY, close with SELL at BID price
                            # If original was SELL, close with BUY at ASK price
                            if mt5_trade_type_for_closure == mt5.ORDER_TYPE_BUY:
                                close_price = tick_info.bid
                            elif mt5_trade_type_for_closure == mt5.ORDER_TYPE_SELL:
                                close_price = tick_info.ask
                            else:
                                print(f"Warning: Unknown position type {mt5_trade_type_for_closure} for ticket {ticket}. Skipping closure.")
                                continue

                            # Create the request to close the position
                            request_close = {
                                "action": mt5.TRADE_ACTION_DEAL,
                                "symbol": symbol,
                                "volume": volume,
                                "type": mt5.ORDER_TYPE_SELL if mt5_trade_type_for_closure == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                                "position": ticket,
                                "price": close_price,
                                "deviation": 20, # Slippage in points, adjust as needed
                                "magic": position.magic, # Use original magic number
                                "comment": "AI Auto-Close",
                                "type_time": mt5.ORDER_TIME_GTC,
                                "type_filling": mt5.ORDER_FILLING_FOK, # Fill or Kill
                            }

                            result = mt5.OrderSend(request_close)

                            if result.retcode == mt5.TRADE_RETCODE_DONE:
                                print(f"*** SUCCESS: Position {ticket} ({symbol} {trade_type_str}) closed by AI. ***")
                            else:
                                print(f"!!! FAILED to close position {ticket} ({symbol} {trade_type_str}).")
                                print(f"    Return code: {result.retcode}")
                                print(f"    Description: {mt5.last_error()}")
                                if result.comment:
                                    print(f"    Comment: {result.comment}")
                        else:
                            print(f"AI did not recommend closing for position {ticket}. No action taken.")

                    else:
                        print("AI advice function (get_gemini_trade_advice) not available.")
                else:
                    print("Elaboration function (elaborate_market_conditions) not available.")
                    print("Raw predictions at trade time:")
                    print(json.dumps(trade_predictions_at_entry, indent=4))
            else:
                print(f"No prediction record found for Magic Number {magic_number} in {ALL_TRADING_RECORDS_FILE}.")
    else:
        print("No open trades found in MetaTrader5.")

    # 4. Provide AI advice for the latest market view (for potential new trades)
    print("\n--- AI Advice for Latest Market View (Potential New Trade) ---")
    if get_gemini_trade_advice:
        latest_timestamp = records['prediction_logs'][-1].get('timestamp', 'N/A')
        ai_prompt_new_trade = (
            f"Act as a seasoned forex trading advisor. You are analyzing the *current* market conditions ({latest_timestamp}). "
            f"Below are the elaborated conditions, including a consolidated summary. "
            f"Based *solely* on these conditions, if you were to open a new trade, would this be a 'BUY opportunity', 'SELL opportunity', or 'NO TRADE'? "
            f"Provide a brief, concise reason (1-2 sentences) for your recommendation.\n\n"
            f"--- Elaborated Market Conditions for Latest View ({latest_timestamp}) ---\n"
            f"{json.dumps(elaborated_latest_conditions, indent=4)}\n\n"
            f"Your Decision for a NEW TRADE (BUY opportunity / SELL opportunity / NO TRADE) and Reasoning:"
        )
        ai_advice_new_trade = get_gemini_trade_advice(ai_prompt_new_trade)
        print(f"AI Advisor's Recommendation for Latest Market View: {ai_advice_new_trade}")
    else:
        print("AI advice function (get_gemini_trade_advice) not available.")

    print("\n--- AI-Powered Analysis Cycle Complete ---")


if __name__ == "__main__":
    # --- Entry Point for continuous operation ---
    if not mt5.initialize():
        print("Failed to initialize MetaTrader 5. Exiting.")
        exit()

    last_run_minute = -1

    try:
        print("Starting AI-Powered Trading Advisor in continuous mode. Analysis every 15 minutes.")
        print("\n*** WARNING: AUTOMATED TRADE CLOSURE IS ENABLED. USE WITH EXTREME CAUTION. ***\n")
        while True:
            now = datetime.now()
            current_minute = now.minute

            target_minutes = [0, 15, 30, 45]

            next_target_minute = None
            for tm in target_minutes:
                if tm >= current_minute:
                    next_target_minute = tm
                    break
            
            if next_target_minute is None:
                next_target_minute = target_minutes[0]
                next_target_hour = (now.hour + 1) % 24
            else:
                next_target_hour = now.hour
            
            next_run_time = now.replace(
                hour=next_target_hour,
                minute=next_target_minute,
                second=1, # Aim for 1 second past the minute
                microsecond=0
            )
            
            if next_run_time <= now:
                minutes_to_add = 15 - (now.minute % 15)
                if minutes_to_add == 15 and now.second > 1:
                    minutes_to_add = 15
                elif minutes_to_add == 15 and now.second <= 1:
                    minutes_to_add = 0
                
                if minutes_to_add == 0 and now.second > 1:
                    minutes_to_add = 15
                
                next_run_time = now + timedelta(minutes=minutes_to_add)
                next_run_time = next_run_time.replace(second=1, microsecond=0)

                if next_run_time.minute not in target_minutes and next_run_time.minute % 15 != 0:
                    next_run_time = next_run_time.replace(minute=(next_run_time.minute // 15 + 1) * 15)
                    next_run_time = next_run_time.replace(second=1, microsecond=0)
                
                if next_run_time <= now:
                    next_run_time += timedelta(minutes=15)
                    next_run_time = next_run_time.replace(second=1, microsecond=0)


            sleep_seconds = (next_run_time - now).total_seconds()
            
            if sleep_seconds < 0:
                sleep_seconds = 0

            print(f"Next analysis scheduled for {next_run_time.strftime('%H:%M:%S')}. Sleeping for {sleep_seconds:.1f} seconds.")
            time.sleep(sleep_seconds)

            now_after_sleep = datetime.now()
            if (now_after_sleep.minute in target_minutes and now_after_sleep.second == 1) and \
               (last_run_minute != now_after_sleep.minute):
                
                analyze_trades_with_ai_guidance()
                last_run_minute = now_after_sleep.minute
            else:
                print(f"Skipping analysis run at {now_after_sleep.strftime('%H:%M:%S')}. Not exactly on interval or already ran.")

            time.sleep(1)

    except KeyboardInterrupt:
        print("AI advisor process interrupted by user (Ctrl+C). Shutting down.")
    except Exception as e:
        print(f"A critical error occurred in the AI advisor process: {e}")
    finally:
        print("Shutting down MetaTrader5.")
        mt5.shutdown()
