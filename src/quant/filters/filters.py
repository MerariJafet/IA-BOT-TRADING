import pandas as pd


class FilterConfig:
    def __init__(
        self,
        max_spread: float = 0.0004,  # 0.04%
        max_volatility: float = 0.005,  # 0.5%
        min_prob: float = 0.55,
    ) -> None:
        self.max_spread = max_spread
        self.max_volatility = max_volatility
        self.min_prob = min_prob


class QuantFilters:
    """Agrupa filtros cuantitativos que bloquean trades de baja calidad."""

    def __init__(self, config: FilterConfig) -> None:
        self.cfg = config

    def spread_ok(self, df: pd.DataFrame) -> bool:
        last = df.iloc[-1]
        spread = (last.ask - last.bid) / last.mid
        return spread <= self.cfg.max_spread

    def volatility_ok(self, df: pd.DataFrame) -> bool:
        last = df.iloc[-1]
        return last.micro_vol <= self.cfg.max_volatility

    def noise_ok(self, df: pd.DataFrame, state: str) -> bool:
        last = df.iloc[-1]
        momo = last.micro_momo

        if state == "UP":
            return momo > 0
        if state == "DOWN":
            return momo < 0
        return False

    def probability_ok(self, prob_up: float, prob_down: float, state: str) -> bool:
        if state == "UP":
            return prob_up >= self.cfg.min_prob
        if state == "DOWN":
            return prob_down >= self.cfg.min_prob
        return False

    def validate(
        self,
        df: pd.DataFrame,
        state: str,
        prob_up: float,
        prob_down: float,
    ) -> tuple[bool, str]:
        """Convenience wrapper returning pass/fail status and reason."""
        return self.apply_all_filters(df, state, prob_up, prob_down)

    def apply_all_filters(
        self,
        df: pd.DataFrame,
        state: str,
        prob_up: float,
        prob_down: float,
    ) -> tuple[bool, str]:
        if not self.spread_ok(df):
            return False, "BAD_SPREAD"

        if not self.volatility_ok(df):
            return False, "HIGH_VOLATILITY"

        if not self.noise_ok(df, state):
            return False, "NOISE"

        if not self.probability_ok(prob_up, prob_down, state):
            return False, "LOW_PROBABILITY"

        return True, "OK"
