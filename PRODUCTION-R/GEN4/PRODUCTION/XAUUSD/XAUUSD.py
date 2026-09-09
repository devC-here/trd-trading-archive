from deploy import TradingPredictor
from datetime import datetime
import time
from placer import execute_automated_trade

def trigger_functions():
    predictor = TradingPredictor()  # Initialize predictor once

    while True:
        now = datetime.now()
        current_time = now.strftime('%H:%M:%S')

        try:
            if current_time.endswith('00:02'):
                prediction_30M = predictor.get_predictions_30M()
                if prediction_30M and "decision" in prediction_30M:
                    execute_automated_trade(prediction_30M["decision"])

            elif current_time.endswith('15:02'):
                prediction_15M = predictor.get_predictions_15M()
                if prediction_15M and "decision" in prediction_15M:
                    execute_automated_trade(prediction_15M["decision"])

            elif current_time.endswith('30:02'):
                prediction_1H = predictor.get_predictions_1H()
                if prediction_1H and "decision" in prediction_1H:
                    execute_automated_trade(prediction_1H["decision"])

            elif current_time.endswith('45:02'):
                prediction_15M = predictor.get_predictions_15M()
                if prediction_15M and "decision" in prediction_15M:
                    execute_automated_trade(prediction_15M["decision"])
        
        except Exception as e:
            print(f"[{datetime.now()}] Error: {e}")

        time.sleep(0.9)  # Reduce CPU usage while maintaining frequent checks

if __name__ == "__main__":
    trigger_functions()
