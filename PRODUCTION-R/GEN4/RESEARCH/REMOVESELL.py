import pandas as pd

def filter_trade_signals(file_path, signal_to_remove, num_remove, output_file):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Map trade decisions to numerical values
    decision_map = {"Sell": 0, "No Trade": 1, "Buy": 2}
    df['trade_decision'] = df['trade_decision'].map(decision_map)
    
    if signal_to_remove not in decision_map:
        raise ValueError("Invalid signal. Choose from: 'Sell', 'No Trade', or 'Buy'")
    
    # Identify the signal to filter
    signal_value = decision_map[signal_to_remove]
    signal_df = df[df['trade_decision'] == signal_value]
    non_signal_df = df[df['trade_decision'] != signal_value]
    
    # Ensure num_remove does not exceed available samples
    num_remove = min(num_remove, len(signal_df))
    
    # Randomly sample signals to remove
    signal_df_filtered = signal_df.sample(n=len(signal_df) - num_remove, random_state=42)
    
    # Combine back and shuffle
    filtered_df = pd.concat([signal_df_filtered, non_signal_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Map back numerical values to trade decision labels
    reverse_map = {0: "Sell", 1: "No Trade", 2: "Buy"}
    filtered_df['trade_decision'] = filtered_df['trade_decision'].map(reverse_map)
    
    # Save the filtered dataset
    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered dataset saved to {output_file}")

# Example usage
filter_trade_signals("trade_decisions2.csv", signal_to_remove="No Trade", num_remove=900, output_file="trade_decisions2.csv")
