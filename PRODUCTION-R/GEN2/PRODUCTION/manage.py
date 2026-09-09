import MetaTrader5 as mt5
from PyQt5 import QtWidgets
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QTimer
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

# Initialize MetaTrader 5pyqt5 ,
if not mt5.initialize():
    QtWidgets.QMessageBox.critical(None, "Initialization Error", "MetaTrader 5 initialization failed")
    sys.exit()

# Global Variables for Trade Manager
SYMBOL = "GBPJPY.0"  # Replace with your trading symbol
TARGET_PRICE = None
THRESHOLD_PERCENTAGE = None
PROFIT_THRESHOLD = None
LOSS_THRESHOLD = None
TRADE_DIRECTION = "Buy"  # Replace with "Sell" if necessary
TRAILING_PROFIT = None
ENABLE_TRAILING = False
HIGHEST_PROFIT = 0.0

# Function to close all trades
def close_all_trades():
    """Closes all open trades."""
    positions = mt5.positions_get()
    if positions is None or len(positions) == 0:
        print("No open trades to close.")
        return

    for pos in positions:
        # Determine the type of order (Buy or Sell) and close accordingly
        order_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(pos.symbol).bid if pos.type == 0 else mt5.symbol_info_tick(pos.symbol).ask

        # Send the close order
        result = mt5.order_send({
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": pos.ticket,
            "price": price,
            "deviation": 10,
        })

        # Check if the trade was successfully closed
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"Closed trade: {pos.ticket}")
        else:
            print(f"Failed to close trade: {pos.ticket}, Error: {result.comment}")

# Combined GUI for Trade Manager and Bulk Trade Placer
class MetaTraderApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("MetaTrader 5 Manager")
        self.setGeometry(100, 100, 300, 600)

        # Main Layout with Tabs
        main_layout = QtWidgets.QVBoxLayout()

        # Tab Widget to hold both functionalities
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(TradeManager(), "Trade Manager")
        self.tabs.addTab(BulkTradePlacer(), "Bulk Trade Placer")
        
        main_layout.addWidget(self.tabs)
        
        self.setLayout(main_layout)

# Trade Manager Tab
class TradeManager(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout()

        # Labels
        self.balance_label = QtWidgets.QLabel("Account Balance: 0.00")
        layout.addWidget(self.balance_label)

        self.profit_label = QtWidgets.QLabel("Current Profit: 0.00")
        layout.addWidget(self.profit_label)

        self.target_label = QtWidgets.QLabel("Target Profit/Loss for Threshold: 0.00")
        layout.addWidget(self.target_label)

        # Closing condition dropdown
        self.closing_condition_label = QtWidgets.QLabel("Closing Condition:")
        self.closing_condition_dropdown = QtWidgets.QComboBox()
        self.closing_condition_dropdown.addItems(["Price", "Percentage", "Profit"])
        layout.addWidget(self.closing_condition_label)
        layout.addWidget(self.closing_condition_dropdown)

        # Inputs for Target Price and Thresholds
        self.target_price_entry = QtWidgets.QLineEdit()
        self.target_price_entry.setPlaceholderText("Enter Target Price")
        layout.addWidget(self.target_price_entry)

        self.set_price_button = QtWidgets.QPushButton("Set Target Price")
        self.set_price_button.clicked.connect(self.set_target_price)
        layout.addWidget(self.set_price_button)

        self.threshold_entry = QtWidgets.QLineEdit()
        self.threshold_entry.setPlaceholderText("Enter % Threshold for Profit/Loss")
        layout.addWidget(self.threshold_entry)

        self.set_threshold_button = QtWidgets.QPushButton("Set % Threshold")
        self.set_threshold_button.clicked.connect(self.set_threshold_percentage)
        layout.addWidget(self.set_threshold_button)

        self.profit_threshold_entry = QtWidgets.QLineEdit()
        self.profit_threshold_entry.setPlaceholderText("Enter Profit Threshold")
        layout.addWidget(self.profit_threshold_entry)

        self.set_profit_button = QtWidgets.QPushButton("Set Profit Threshold")
        self.set_profit_button.clicked.connect(self.set_profit_threshold)
        layout.addWidget(self.set_profit_button)

        self.loss_threshold_entry = QtWidgets.QLineEdit()
        self.loss_threshold_entry.setPlaceholderText("Enter Loss Threshold")
        layout.addWidget(self.loss_threshold_entry)

        self.set_loss_button = QtWidgets.QPushButton("Set Loss Threshold")
        self.set_loss_button.clicked.connect(self.set_loss_threshold)
        layout.addWidget(self.set_loss_button)

        # Trailing Profit Section
        self.trailing_profit_entry = QtWidgets.QLineEdit()
        self.trailing_profit_entry.setPlaceholderText("Enter Trailing Profit Value")
        layout.addWidget(self.trailing_profit_entry)

        self.set_trailing_button = QtWidgets.QPushButton("Set Trailing Profit")
        self.set_trailing_button.clicked.connect(self.set_trailing_profit)
        layout.addWidget(self.set_trailing_button)

        self.trailing_checkbox = QtWidgets.QCheckBox("Enable Profit Trailing")
        self.trailing_checkbox.stateChanged.connect(self.toggle_trailing)
        layout.addWidget(self.trailing_checkbox)

        # Close Trades Button
        self.close_button = QtWidgets.QPushButton("Close All Trades")
        self.close_button.clicked.connect(close_all_trades)
        layout.addWidget(self.close_button)

        # Start Timer for Updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(1000)  # Update every second

        # Set Layout
        self.setLayout(layout)

    def set_target_price(self):
        global TARGET_PRICE
        try:
            TARGET_PRICE = float(self.target_price_entry.text())
            print(f"Target price set to: {TARGET_PRICE}")
        except ValueError:
            print("Invalid target price input.")

    def set_threshold_percentage(self):
        global THRESHOLD_PERCENTAGE
        try:
            THRESHOLD_PERCENTAGE = float(self.threshold_entry.text())
            print(f"Threshold percentage set to: {THRESHOLD_PERCENTAGE}%")
        except ValueError:
            print("Invalid threshold percentage input.")

    def set_profit_threshold(self):
        global PROFIT_THRESHOLD
        try:
            PROFIT_THRESHOLD = float(self.profit_threshold_entry.text())
            print(f"Profit threshold set to: {PROFIT_THRESHOLD}")
        except ValueError:
            print("Invalid profit threshold input.")

    def set_loss_threshold(self):
        global LOSS_THRESHOLD
        try:
            LOSS_THRESHOLD = float(self.loss_threshold_entry.text())
            print(f"Loss threshold set to: {LOSS_THRESHOLD}")
        except ValueError:
            print("Invalid loss threshold input.")

    def set_trailing_profit(self):
        global TRAILING_PROFIT
        try:
            TRAILING_PROFIT = float(self.trailing_profit_entry.text())
            print(f"Trailing profit set to: {TRAILING_PROFIT}")
        except ValueError:
            print("Invalid trailing profit input.")

    def toggle_trailing(self, state):
        global ENABLE_TRAILING
        ENABLE_TRAILING = bool(state)
        print(f"Profit trailing {'enabled' if ENABLE_TRAILING else 'disabled'}.")

    def update_data(self):
        global TARGET_PRICE, THRESHOLD_PERCENTAGE, PROFIT_THRESHOLD, LOSS_THRESHOLD, TRAILING_PROFIT, ENABLE_TRAILING, HIGHEST_PROFIT
        account_info = mt5.account_info()
        if account_info:
            account_balance = account_info.balance
            self.balance_label.setText(f"Account Balance: {account_balance:.2f}")

            # Calculate and display target profit/loss
            target_profit_loss = (account_balance * THRESHOLD_PERCENTAGE) / 100 if THRESHOLD_PERCENTAGE else 0.0
            self.target_label.setText(f"Target Profit/Loss for Threshold: {target_profit_loss:.2f}")

        total_profit = 0
        for pos in mt5.positions_get():
            total_profit += pos.profit

        self.profit_label.setText(f"Current Profit: {total_profit:.2f}")

        # Update highest profit for trailing
        if ENABLE_TRAILING and total_profit > HIGHEST_PROFIT:
            HIGHEST_PROFIT = total_profit
            print(f"Highest profit updated to: {HIGHEST_PROFIT:.2f}")

        if ENABLE_TRAILING and TRAILING_PROFIT and total_profit <= HIGHEST_PROFIT - TRAILING_PROFIT:
            print(f"Trailing profit condition met. Closing all trades.")
            close_all_trades()

        if PROFIT_THRESHOLD and total_profit >= PROFIT_THRESHOLD:
            print(f"Profit threshold reached: {total_profit:.2f}. Closing all trades.")
            close_all_trades()

        if LOSS_THRESHOLD and total_profit <= -LOSS_THRESHOLD:
            print(f"Loss threshold reached: {total_profit:.2f}. Closing all trades.")
            close_all_trades()

        if self.closing_condition_dropdown.currentText() == "Price" and TARGET_PRICE:
            symbol_info_tick = mt5.symbol_info_tick(SYMBOL)
            if symbol_info_tick:
                current_price = symbol_info_tick.ask
                if current_price >= TARGET_PRICE:
                    print(f"Target price reached: {current_price}. Closing all trades.")
                    close_all_trades()

# Bulk Trade Placer Tab
class BulkTradePlacer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QGridLayout()

        label_font = QFont("Arial", 10)

        # Add Widgets to the Layout
        symbol_label = QtWidgets.QLabel("Symbol")
        symbol_label.setFont(label_font)
        layout.addWidget(symbol_label, 0, 0)
        self.symbol_input = QtWidgets.QLineEdit("XAUUSD")
        layout.addWidget(self.symbol_input, 0, 1)

        order_type_label = QtWidgets.QLabel("Order Type")
        order_type_label.setFont(label_font)
        layout.addWidget(order_type_label, 1, 0)
        self.order_type_buy = QtWidgets.QRadioButton("Buy")
        self.order_type_sell = QtWidgets.QRadioButton("Sell")
        self.order_type_buy.setChecked(True)
        layout.addWidget(self.order_type_buy, 1, 1)
        layout.addWidget(self.order_type_sell, 1, 2)

        volume_label = QtWidgets.QLabel("Volume")
        volume_label.setFont(label_font)
        layout.addWidget(volume_label, 2, 0)
        self.volume_input = QtWidgets.QLineEdit("50")
        layout.addWidget(self.volume_input, 2, 1)

        entry_price_label = QtWidgets.QLabel("Entry Price")
        entry_price_label.setFont(label_font)
        layout.addWidget(entry_price_label, 3, 0)
        self.entry_price_input = QtWidgets.QLineEdit("Market")
        layout.addWidget(self.entry_price_input, 3, 1)

        sl_label = QtWidgets.QLabel("Stop Loss")
        sl_label.setFont(label_font)
        layout.addWidget(sl_label, 4, 0)
        self.sl_input = QtWidgets.QLineEdit("0")
        layout.addWidget(self.sl_input, 4, 1)

        tp_label = QtWidgets.QLabel("Take Profit")
        tp_label.setFont(label_font)
        layout.addWidget(tp_label, 5, 0)
        self.tp_input = QtWidgets.QLineEdit("0")
        layout.addWidget(self.tp_input, 5, 1)

        num_trades_label = QtWidgets.QLabel("Number of Trades")
        num_trades_label.setFont(label_font)
        layout.addWidget(num_trades_label, 6, 0)
        self.num_trades_input = QtWidgets.QLineEdit("60")
        layout.addWidget(self.num_trades_input, 6, 1)

        # Place Trades Button
        self.place_trades_button = QtWidgets.QPushButton("Place Trades")
        self.place_trades_button.setFont(QFont("Arial", 12))
        self.place_trades_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px;")
        self.place_trades_button.clicked.connect(self.place_trades)
        layout.addWidget(self.place_trades_button, 7, 1, 1, 2)

        # Apply Layout
        self.setLayout(layout)

    def place_trades(self):
        symbol = self.symbol_input.text().strip()
        order_type = "Buy" if self.order_type_buy.isChecked() else "Sell"
        volume = float(self.volume_input.text())
        entry_price_text = self.entry_price_input.text().strip()
        sl = float(self.sl_input.text()) if self.sl_input.text() else 0
        tp = float(self.tp_input.text()) if self.tp_input.text() else 0
        num_trades = int(self.num_trades_input.text())

        # Verify Symbol Information
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            QtWidgets.QMessageBox.critical(None, "Symbol Error", f"Symbol '{symbol}' not found in MetaTrader 5.")
            return

        # Confirm before placing multiple trades
        confirm = QtWidgets.QMessageBox.question(
            None, "Confirm Trades",
            f"Are you sure you want to place {num_trades} {order_type} trades for {symbol}?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
        )
        if confirm != QtWidgets.QMessageBox.Yes:
            return

        # Determine MetaTrader 5 order type
        order_type_mt5 = mt5.ORDER_TYPE_BUY if order_type == "Buy" else mt5.ORDER_TYPE_SELL

        # Determine if using market price
        if entry_price_text.lower() == "market":
            entry_price = None
        else:
            try:
                entry_price = float(entry_price_text)
            except ValueError:
                QtWidgets.QMessageBox.critical(None, "Entry Price Error", "Invalid entry price format")
                return

        # Create a ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            futures = []
            for _ in range(num_trades):
                price = entry_price or (mt5.symbol_info_tick(symbol).ask if order_type == "Buy" else mt5.symbol_info_tick(symbol).bid)
                futures.append(executor.submit(self.place_single_trade, symbol, order_type_mt5, volume, price, sl, tp))

            # Collect and handle results
            successful_trades = 0
            for future in as_completed(futures):
                result = future.result()
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    successful_trades += 1
                else:
                    QtWidgets.QMessageBox.warning(None, "Trade Error", f"Failed to place trade: {result.comment}")

        # Show final result
        QtWidgets.QMessageBox.information(None, "Trade Summary", f"Successfully placed {successful_trades} out of {num_trades} trades.")

    def place_single_trade(self, symbol, order_type_mt5, volume, price, sl, tp):
        order = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type_mt5,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 234000,
            "comment": "Bulk Order",
        }
        return mt5.order_send(order)


# Run the application
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MetaTraderApp()
    window.show()
    sys.exit(app.exec_())
