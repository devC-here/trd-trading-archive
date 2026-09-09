from smclib import connect, disconnect, get_eurusd_1h, build_features

def main():
    # 1. Connect to MT5
    connect()

    try:
        # 2. Fetch data (last 200 H1 candles)
        df = get_eurusd_1h(200)

        print("\n=== RAW DATA ===")
        print(df.tail())

        # 3. Build features (SMC pipeline)
        features = build_features(df)

        print("\n=== FEATURES (last rows) ===")
        print(features.tail())

        # 4. Example: print only signals
        print("\n=== BOS / CHOCH ===")
        print(features[['close', 'signal', 'signal_type', 'signal_dir']].dropna().tail())

    finally:
        # 5. Always disconnect
        disconnect()


if __name__ == "__main__":
    main()