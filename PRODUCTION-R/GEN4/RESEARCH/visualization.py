import sys
import pandas as pd
import matplotlib
matplotlib.use("QtAgg")  # Ensure compatibility with PyQt6
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtCore import Qt

class CandleViewer(QWidget):
    def __init__(self):
        super().__init__()
        
        # Load the dataset
        self.df = pd.read_csv("modified_trade_decisions.csv", parse_dates=["time"])
        self.index = 0
        self.color_map = {"Buy": "green", "Sell": "red", "No Trade": "gray"}
        
        # Enable keyboard focus
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # Layout setup
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Matplotlib figure
        self.figure, self.axes = plt.subplots(1, 2, figsize=(12, 5))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        # Navigation buttons
        self.btn_prev = QPushButton("Previous")
        self.btn_next = QPushButton("Next")
        self.btn_buy = QPushButton("Buy")
        self.btn_sell = QPushButton("Sell")
        self.btn_no_trade = QPushButton("No Trade")
        
        self.btn_prev.clicked.connect(self.prev_candle)
        self.btn_next.clicked.connect(self.next_candle)
        self.btn_buy.clicked.connect(lambda: self.change_decision("Buy"))
        self.btn_sell.clicked.connect(lambda: self.change_decision("Sell"))
        self.btn_no_trade.clicked.connect(lambda: self.change_decision("No Trade"))
        
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        self.layout.addLayout(nav_layout)
        
        decision_layout = QHBoxLayout()
        decision_layout.addWidget(self.btn_buy)
        decision_layout.addWidget(self.btn_sell)
        decision_layout.addWidget(self.btn_no_trade)
        self.layout.addLayout(decision_layout)
        
        # Save button
        self.btn_save = QPushButton("Save Changes")
        self.btn_save.clicked.connect(self.save_changes)
        self.layout.addWidget(self.btn_save)
        
        self.plot_candles()
    
    def plot_candles(self):
        if self.index >= len(self.df) - 1:
            return
        
        self.axes[0].clear()
        self.axes[1].clear()
        
        row_current = self.df.iloc[self.index]
        row_next = self.df.iloc[self.index + 1]
        
        for ax, row, title in zip(self.axes, [row_current, row_next], ["Current Candle", "Next Candle"]):
            ax.plot([1, 1], [row['low'], row['high']], color='black')
            ax.bar(1, row['close'] - row['open'],
                   bottom=min(row['open'], row['close']),
                   color='green' if row['close'] > row['open'] else 'red',
                   width=0.3)
            ax.set_title(f"{title}\nTrade Decision: {row['trade_decision']}")
            ax.set_xlim(0, 2)
            ax.set_xticks([])
        
        self.canvas.draw()
    
    def next_candle(self):
        if self.index < len(self.df) - 2:
            self.index += 1
            self.plot_candles()
    
    def prev_candle(self):
        if self.index > 0:
            self.index -= 1
            self.plot_candles()
    
    def change_decision(self, decision):
        self.df.at[self.index, 'trade_decision'] = decision
        self.plot_candles()
    
    def save_changes(self):
        self.df.to_csv("modified_trade_decisions.csv", index=False)
        print("Changes saved!")
    
    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Right:  # Right Arrow (Page Right)
            self.next_candle()
        elif event.key() == Qt.Key.Key_Left:  # Left Arrow (Page Left)
            self.prev_candle()
        super().keyPressEvent(event)  # Ensure the base class processes other keys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = CandleViewer()
    viewer.show()
    sys.exit(app.exec())
