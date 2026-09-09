import pandas as pd

# Load the dataset
data_file = "data.txt"
try:
    # Read the dataset
    with open(data_file, "r") as file:
        lines = file.readlines()

    # Extract the header and data
    raw_header = lines[0].strip()
    raw_data = lines[1:]

    # Process the header to split columns correctly
    header = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume", "identification"]

    # Prepare the data by splitting based on fixed columns
    data = []
    for line in raw_data:
        parts = line.strip().split(maxsplit=8)  # Split into 9 parts (8 splits for 9 columns)
        data.append(parts)

    # Create DataFrame
    df = pd.DataFrame(data, columns=header)

    # Convert appropriate columns to numeric types
    numeric_columns = ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Display the cleaned dataset
    print("Cleaned dataset:")
    print(df.head())

except Exception as e:
    print(f"Error processing the dataset: {e}")
