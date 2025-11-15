"""Route raw market data through the quant pipeline to obtain trade intents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from src.quant.filters.filters import FilterConfig, QuantFilters
from src.quant.micro.microstructure import MicrostructureConfig, MicrostructureModel
from src.quant.signals.signal_engine import SignalEngine, SignalThresholds


@dataclass
class RouterResult:
	signal: str
	prob_up: float
	prob_down: float
	micro_state: str
	filters_passed: bool
	filters_reason: str

	def as_dict(self) -> Dict[str, Any]:
		return {
			"signal": self.signal,
			"prob_up": self.prob_up,
			"prob_down": self.prob_down,
			"micro_state": self.micro_state,
			"filters_passed": self.filters_passed,
			"filters_reason": self.filters_reason,
		}


class SignalRouter:
	"""Coordinate signal engine, microstructure model, and quantitative filters."""

	def __init__(
		self,
		thresholds: SignalThresholds | None = None,
		micro_config: MicrostructureConfig | None = None,
		filter_config: FilterConfig | None = None,
	) -> None:
		self.engine = SignalEngine(thresholds)
		self.micro = MicrostructureModel(micro_config)
		self.filters = QuantFilters(filter_config or FilterConfig())

	def route(self, df: pd.DataFrame) -> RouterResult:
		if df.empty:
			raise ValueError("SignalRouter.route received empty dataframe")

		signal = self.engine.generate_signal(df)
		prob_up, prob_down, micro_state = self.micro.get_direction_probability(df)
		enriched = self.micro.compute_features(df)
		filters_pass, reason = self.filters.apply_all_filters(
			enriched,
			micro_state,
			prob_up,
			prob_down,
		)

		if not filters_pass:
			return RouterResult(
				signal="NONE",
				prob_up=prob_up,
				prob_down=prob_down,
				micro_state=micro_state,
				filters_passed=False,
				filters_reason=reason,
			)

		confirmed_signal = signal
		if signal == "BUY" and micro_state != "UP":
			confirmed_signal = "NONE"
		if signal == "SELL" and micro_state != "DOWN":
			confirmed_signal = "NONE"

		return RouterResult(
			signal=confirmed_signal,
			prob_up=prob_up,
			prob_down=prob_down,
			micro_state=micro_state,
			filters_passed=True,
			filters_reason="OK",
		)
