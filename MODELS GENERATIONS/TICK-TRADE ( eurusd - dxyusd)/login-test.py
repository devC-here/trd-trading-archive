import MetaTrader5 as mt5
import time

# --- CONFIGURATION ---
# Replace with your actual credentials
ACCOUNTS = [
    {"login": 168348, "password": "yc=!rtk3xX", "server": "AudaCityGlobal-Live"},
    {"login": 169283, "password": "YU=2oVvHnL", "server": "AudaCityGlobal-Live"},
    {"login": 169350, "password": "O!9IsFqwCr", "server": "AudaCityGlobal-Live"}
]

def benchmark_switching():
    if not mt5.initialize():
        print("MT5 Initialization failed")
        return

    results = []

    print(f"{'Account':<15} | {'Server':<20} | {'Switch Time (sec)':<15}")
    print("-" * 55)

    for acc in ACCOUNTS:
        start_time = time.perf_counter()
        
        # The actual switch happens here
        authorized = mt5.login(
            login=acc['login'], 
            password=acc['password'], 
            server=acc['server']
        )
        
        end_time = time.perf_counter()
        duration = round(end_time - start_time, 4)

        if authorized:
            status = f"{duration}s"
            results.append(duration)
        else:
            status = f"FAILED ({mt5.last_error()})"
        
        print(f"{acc['login']:<15} | {acc['server']:<20} | {status}")
        
        # Small sleep to let the terminal UI "breathe" before the next jump
        time.sleep(1)

    if results:
        avg = round(sum(results) / len(results), 4)
        print("-" * 55)
        print(f"Average Transition Time: {avg} seconds")

    mt5.shutdown()

if __name__ == "__main__":
    benchmark_switching()