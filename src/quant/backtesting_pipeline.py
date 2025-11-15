from dataclasses import dataclass
import pandas as pd
import numpy as np
from typing import List, Dict, Any

@dataclass
class BacktestConfig:
    initial_balance: float = 1000.0
    fee_rate: float = 0.0004
    slippage: float = 1.0
    risk_per_trade: float = 0.02
    verbose: bool = False
    stop_loss_pct: float = 0.003
    take_profit_pct: float = 0.006

class BacktestingPipeline:

    def __init__(self, config: BacktestConfig):
        self.cfg = config
        self.balance = config.initial_balance
        self.position = None
        self.entry_price = None
        self.equity = [self.balance]
        self.trades: List[Dict[str,Any]] = []

    def ingest_data(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        return self

    def apply_strategy(self, strategy_fn):
        self.strategy_fn = strategy_fn
        return self

    def run(self):
        for _, row in self.df.iterrows():
            price = row["close"]
            signal = self.strategy_fn(row)

            self._process_signal(signal, price)
            self.equity.append(self.balance)

        return self._results()

    def _process_signal(self, signal: str, price: float):

        if self.position is not None:
            self._check_sl_tp(price)

        if signal == "BUY" and self.position is None:
            size = self._position_size(price)
            self.position = size
            self.entry_price = price

        elif signal == "SELL" and self.position is not None:
            self._close_position(price, reason="SELL signal")

    def _check_sl_tp(self, price):
        change = (price - self.entry_price) / self.entry_price

        if change <= -self.cfg.stop_loss_pct:
            self._close_position(price, reason="STOP LOSS")

        elif change >= self.cfg.take_profit_pct:
            self._close_position(price, reason="TAKE PROFIT")

    def _position_size(self, price):
        risk_amount = self.balance * self.cfg.risk_per_trade
        size = risk_amount / (price * self.cfg.stop_loss_pct)
        return max(size, 0)

    def _close_position(self, price, reason: str):
        pnl = (price - self.entry_price) * self.position
        pnl -= abs(price * self.cfg.fee_rate)
        pnl -= self.cfg.slippage

        self.balance += pnl
        self.trades.append({"pnl": pnl, "reason": reason})

        self.position = None
        self.entry_price = None

    def _results(self):
        df_eq = pd.Series(self.equity)

        returns = df_eq.pct_change().dropna()

        sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
        mdd = (df_eq / df_eq.cummax() - 1).min()

        winrate = np.mean([t["pnl"] > 0 for t in self.trades]) if self.trades else 0.0
        profit_factor = (sum(t["pnl"] for t in self.trades if t["pnl"] > 0) /
                         abs(sum(t["pnl"] for t in self.trades if t["pnl"] < 0) + 1e-9))

        return {
            "final_balance": self.balance,
            "trades": self.trades,
            "sharpe": float(sharpe),
            "max_drawdown": float(mdd),
            "winrate": float(winrate),
            "profit_factor": float(profit_factor),
            "equity_curve": df_eq.tolist()
        }
