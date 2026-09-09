from deploy import TradingPredictor
from datetime import datetime
import time
from placer import execute_automated_trade

def trigger_functions():
    predictor = TradingPredictor()  # Initialize predictor once
    
    while True:
        now = datetime.now()
        current_time = now.strftime('%H:%M:%S')

        if current_time.endswith('00:02'):
            prediction_30M = predictor.get_predictions_30M()
            if prediction_30M:  # Ensure we have a valid prediction
                execute_automated_trade(prediction_30M["decision"])  # Extract only the decision string
            time.sleep(1.5)  

        elif current_time.endswith('30:02'):
            prediction_1H = predictor.get_predictions_1H()
            if prediction_1H:  # Ensure we have a valid prediction
                execute_automated_trade(prediction_1H["decision"])  # Extract only the decision string
            time.sleep(1.5)  
        
        time.sleep(0.5)  # Reduce CPU usage while still checking frequently

if __name__ == "__main__":
    trigger_functions()
 