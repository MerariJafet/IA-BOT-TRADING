"""Signal engine module computing microstructure-driven signals."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SignalThresholds:
    spread_threshold: float = 0.00020  # 2 bps
    vol_window: int = 60
    vol_threshold: float = 0.0025  # 0.25%
    ema_fast: int = 20
    ema_slow: int = 60
    imbalance_threshold: float = 0.55


class SignalEngine:
    """Generate trading signals using microstructure-derived context."""

    def __init__(self, thresholds: SignalThresholds | None = None) -> None:
        self.thresholds = thresholds or SignalThresholds()

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add spread, volatility, momentum, EMA, and order-imbalance columns."""
        df = df.copy()
        df["spread_pct"] = (df["ask"] - df["bid"]) / df["mid"]
        df["returns"] = df["mid"].pct_change()
        df["volatility"] = df["returns"].rolling(self.thresholds.vol_window).std().fillna(0)
        df["micro_momo"] = df["mid"].diff().fillna(0)
        df["ema_fast"] = df["mid"].ewm(span=self.thresholds.ema_fast, adjust=False).mean()
        df["ema_slow"] = df["mid"].ewm(span=self.thresholds.ema_slow, adjust=False).mean()
        df["ema_signal"] = df["ema_fast"] - df["ema_slow"]
        denom = df["volume_buy"] + df["volume_sell"] + 1e-9
        df["imbalance"] = df["volume_buy"] / denom
        return df

    def generate_signal(self, df: pd.DataFrame) -> str:
        """Return BUY/SELL/NONE based on the most recent sample."""
        enriched = self.compute_features(df)
        last = enriched.iloc[-1]

        if last.spread_pct > self.thresholds.spread_threshold:
            return "NONE"
        if last.volatility > self.thresholds.vol_threshold:
            return "NONE"

        ema_up = last.ema_fast > last.ema_slow
        ema_down = last.ema_fast < last.ema_slow
        prob_up = last.imbalance
        prob_down = 1 - last.imbalance
        momo_up = last.micro_momo > 0
        momo_down = last.micro_momo < 0

        if ema_up and momo_up and prob_up > self.thresholds.imbalance_threshold:
            return "BUY"
        if ema_down and momo_down and prob_down > self.thresholds.imbalance_threshold:
            return "SELL"
        return "NONE"

