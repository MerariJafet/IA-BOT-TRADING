"""Microstructure analytics for probabilistic signal confirmation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class MicrostructureConfig:
	imbalance_threshold: float = 0.55
	momo_window: int = 5
	vol_window: int = 10


class MicrostructureModel:
	"""Compute microstructure features and directional probabilities."""

	def __init__(self, config: MicrostructureConfig | None = None) -> None:
		self.config = config or MicrostructureConfig()

	def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
		"""Calculate imbalance, volume skew, micro-momentum, and micro-vol."""
		df = df.copy()
		denom = df["volume_buy"] + df["volume_sell"] + 1e-9
		df["order_imbalance"] = df["volume_buy"] / denom
		df["vol_imbalance"] = (df["volume_buy"] - df["volume_sell"]) / denom
		df["micro_momo"] = df["mid"].diff(self.config.momo_window).fillna(0)
		df["micro_vol"] = df["mid"].pct_change().rolling(self.config.vol_window).std().fillna(0)
		return df

	def get_direction_probability(self, df: pd.DataFrame) -> tuple[float, float, str]:
		"""Return (prob_up, prob_down, state) using latest computed features."""
		enriched = self.compute_features(df)
		last = enriched.iloc[-1]
		prob_up = last.order_imbalance
		prob_down = 1 - prob_up
		momentum_up = last.micro_momo > 0
		momentum_down = last.micro_momo < 0

		if prob_up >= self.config.imbalance_threshold and momentum_up:
			return prob_up, prob_down, "UP"
		if prob_down >= self.config.imbalance_threshold and momentum_down:
			return prob_up, prob_down, "DOWN"
		return prob_up, prob_down, "NONE"
