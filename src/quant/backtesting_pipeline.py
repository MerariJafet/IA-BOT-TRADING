from dataclasses import dataclass
import pandas as pd
from typing import Optional, Dict, Any

@dataclass
class BacktestConfig:
    initial_balance: float = 1000.0
    fee_rate: float = 0.0004
    slippage: float = 0.5
    risk_per_trade: float = 0.01
    verbose: bool = False

class BacktestDataset:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @staticmethod
    def from_parquet(path: str):
        return BacktestDataset(pd.read_parquet(path))

class BacktestingPipeline:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.balance = config.initial_balance
        self.position = None  # BUY or SELL
        self.entry_price = None
        self.trades = []

    def ingest_data(self, df: pd.DataFrame):
        self.data = df
        return self

    def apply_strategy(self, strategy_fn):
        self.strategy_fn = strategy_fn
        return self

    def run(self):
        for idx, row in self.data.iterrows():
            signal = self.strategy_fn(row)
            self._process_signal(signal, row)
        return self.trades, self.balance

    def _process_signal(self, signal: str, row: pd.Series):
        price = row['close']

        if signal == 'BUY' and self.position is None:
            self.position = 'LONG'
            self.entry_price = price
        elif signal == 'SELL' and self.position == 'LONG':
            pnl = (price - self.entry_price) - (price * self.config.fee_rate)
            self.balance += pnl
            self.trades.append({'side': 'SELL', 'pnl': pnl})
            self.position = None
