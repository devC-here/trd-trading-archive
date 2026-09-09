import sys
import MetaTrader5 as mt5
import pandas as pd
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox
from PyQt6.QtCore import Qt, QObject, QEvent
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Connect to MT5
if not mt5.initialize():
    print("Failed to initialize MT5")
    sys.exit()

symbol = "XAUUSD"  # Using XAUUSD
timeframe = mt5.TIMEFRAME_H1
num_candles = 2100  # Fetching 1000 candles

# Retrieve historical data
data = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)
if data is None:
    print("Failed to retrieve data")
    mt5.shutdown()
    sys.exit()

df = pd.DataFrame(data)
mt5.shutdown()

df['time'] = pd.to_datetime(df['time'], unit='s')

# Custom event filter for key handling
class KeyPressFilter(QObject):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key in {Qt.Key.Key_PageUp, Qt.Key.Key_PageDown}:  # Page Up/Down for "No Trade"
                self.parent.next_candle("No Trade")
                return True
            elif key == Qt.Key.Key_Left:  # Left arrow for "Sell"
                self.parent.next_candle("Sell")
                return True
            elif key == Qt.Key.Key_Right:  # Right arrow for "Buy"
                self.parent.next_candle("Buy")
                return True
        return super().eventFilter(obj, event)

# GUI Application
class CandleTradeClassifier(QWidget):
    def __init__(self):
        super().__init__()
        self.current_index = 0
        self.dataset = []
        self.init_ui()
        self.update_chart()

        # Install event filter globally
        self.key_filter = KeyPressFilter(self)
        QApplication.instance().installEventFilter(self.key_filter)

    def init_ui(self):
        self.setWindowTitle("Candle Trade Decision")
        self.setGeometry(100, 100, 600, 600)

        self.label = QLabel(f"Candle: {self.current_index + 1}/{num_candles}")
        
        self.buy_button = QPushButton("Buy (→)")
        self.buy_button.clicked.connect(lambda: self.next_candle("Buy"))
        
        self.sell_button = QPushButton("Sell (←)")
        self.sell_button.clicked.connect(lambda: self.next_candle("Sell"))
        
        self.no_trade_button = QPushButton("No Trade (PgUp/PgDn)")
        self.no_trade_button.clicked.connect(lambda: self.next_candle("No Trade"))
        
        self.save_button = QPushButton("Save Dataset")
        self.save_button.clicked.connect(self.save_dataset)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.buy_button)
        button_layout.addWidget(self.sell_button)
        button_layout.addWidget(self.no_trade_button)
        button_layout.addWidget(self.save_button)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def update_chart(self):
        self.ax.clear()
        row = df.iloc[self.current_index]

        # Draw candlestick with wick and body
        self.ax.plot([1, 1], [row['low'], row['high']], color='black')  # Wick
        self.ax.bar(1, row['close'] - row['open'], 
                    bottom=min(row['open'], row['close']), 
                    color='green' if row['close'] > row['open'] else 'red', 
                    width=0.3)

        self.ax.set_xlim(0, 2)
        self.ax.set_xticks([])
        self.ax.set_ylabel("Price")
        self.canvas.draw()
        
        self.label.setText(f"Candle: {self.current_index + 1}/{num_candles}")

    def next_candle(self, trade_decision):
        self.dataset.append({
            "time": df.iloc[self.current_index]['time'],
            "open": df.iloc[self.current_index]['open'],
            "high": df.iloc[self.current_index]['high'],
            "low": df.iloc[self.current_index]['low'],
            "close": df.iloc[self.current_index]['close'],
            "volume": df.iloc[self.current_index]['tick_volume'],  # Fixed incorrect column
            "trade_decision": trade_decision
        })

        self.current_index += 1
        if self.current_index < len(df):
            self.update_chart()
        else:
            self.complete_classification()
        
        # **Auto-save dataset to prevent loss**
        self.auto_save_dataset()

    def complete_classification(self):
        """Handles classification completion."""
        self.label.setText("Classification Completed! Click Save Dataset.")
        self.buy_button.setEnabled(False)
        self.sell_button.setEnabled(False)
        self.no_trade_button.setEnabled(False)

        # Show message box to inform user
        msg = QMessageBox(self)
        msg.setWindowTitle("Classification Complete")
        msg.setText("You have classified all candles. The dataset has been auto-saved.")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()

    def save_dataset(self):
        """Saves the classified dataset manually."""
        pd.DataFrame(self.dataset).to_csv("trade_decisions1.csv", index=False)
        self.label.setText("Dataset Saved Successfully!")

        # Show message box to confirm save
        msg = QMessageBox(self)
        msg.setWindowTitle("Dataset Saved")
        msg.setText("The dataset has been successfully saved as 'trade_decisions1.csv'.")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()

    def auto_save_dataset(self):
        """Automatically saves dataset after every classification to prevent data loss."""
        pd.DataFrame(self.dataset).to_csv("trade_decisions1.csv", index=False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CandleTradeClassifier()
    window.show()
    sys.exit(app.exec())
