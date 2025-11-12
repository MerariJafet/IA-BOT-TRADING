"""
Evaluador de Benchmark para comparar performance del sistema híbrido
con benchmarks estándar (BTC/USD, S&P500).

Calcula métricas clave:
- ROI mensualizado
- Alpha (retorno excedente vs benchmark)
- Beta (sensibilidad al benchmark)
- Sharpe Ratio rolling
- Maximum Drawdown
- Correlation con benchmarks

Genera reportes y gráficos de equity curve.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.stats import linregress

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"

LIVE_TRADES_PATH = DATA_DIR / "live_trades.parquet"
BENCHMARK_COMPARISON_PATH = REPORTS_DIR / "benchmark_comparison.json"
EQUITY_CURVE_PATH = REPORTS_DIR / "equity_curve.png"


class BenchmarkEvaluator:
    """
    Evaluador de benchmark que compara el sistema de trading
    con BTC/USD y S&P500.
    """

    def __init__(
        self,
        trades_path: str = str(LIVE_TRADES_PATH),
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.04,  # 4% anual
    ):
        """
        Args:
            trades_path: Ruta al archivo de trades
            initial_capital: Capital inicial para cálculos
            risk_free_rate: Tasa libre de riesgo anual (para Sharpe y Alpha)
        """
        self.trades_path = Path(trades_path)
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate

        self.trades: Optional[pd.DataFrame] = None
        self.btc_prices: Optional[pd.DataFrame] = None
        self.sp500_prices: Optional[pd.DataFrame] = None

    def load_trades(self) -> pd.DataFrame:
        """Carga los trades del sistema."""
        if not self.trades_path.exists():
            raise FileNotFoundError(f"No se encontró archivo de trades: {self.trades_path}")

        self.trades = pd.read_parquet(self.trades_path)

        if "timestamp" in self.trades.columns:
            self.trades["timestamp"] = pd.to_datetime(self.trades["timestamp"])
            self.trades = self.trades.sort_values("timestamp")

        return self.trades

    def fetch_btc_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Descarga precios históricos de BTC/USD desde CoinGecko API.

        Args:
            start_date: Fecha inicial YYYY-MM-DD
            end_date: Fecha final YYYY-MM-DD

        Returns:
            DataFrame con columnas: timestamp, btc_price
        """
        # CoinGecko API - /coins/bitcoin/market_chart/range
        start_ts = int(pd.Timestamp(start_date).timestamp())
        end_ts = int(pd.Timestamp(end_date).timestamp())

        url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
        params = {"vs_currency": "usd", "from": start_ts, "to": end_ts}

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            prices = data.get("prices", [])

            df = pd.DataFrame(prices, columns=["timestamp", "btc_price"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            self.btc_prices = df.sort_values("timestamp")
            return self.btc_prices

        except Exception as e:
            print(f"Error descargando precios de BTC: {e}")
            # Retornar DataFrame vacío
            return pd.DataFrame(columns=["timestamp", "btc_price"])

    def fetch_sp500_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Descarga precios históricos de S&P500 (simulado).

        Nota: Para producción, usar APIs como Alpha Vantage, Yahoo Finance, etc.
        Por ahora, generamos datos sintéticos.

        Args:
            start_date: Fecha inicial YYYY-MM-DD
            end_date: Fecha final YYYY-MM-DD

        Returns:
            DataFrame con columnas: timestamp, sp500_price
        """
        # Generar datos sintéticos para S&P500
        # En producción, usar una API real
        timestamps = pd.date_range(start=start_date, end=end_date, freq="D")

        # Simulamos precio inicial de 4500 con drift alcista moderado y volatilidad baja
        np.random.seed(42)
        initial_price = 4500.0
        returns = np.random.normal(0.0003, 0.01, len(timestamps))  # ~0.03% diario, vol 1%
        prices = initial_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({"timestamp": timestamps, "sp500_price": prices})

        self.sp500_prices = df
        return self.sp500_prices

    def calculate_portfolio_equity_curve(self) -> pd.DataFrame:
        """
        Calcula la equity curve del portfolio basado en los trades.

        Returns:
            DataFrame con: timestamp, equity
        """
        if self.trades is None or self.trades.empty:
            raise ValueError("Trades no cargados. Ejecutar load_trades() primero.")

        # Calcular equity acumulada
        trades_with_equity = self.trades.copy()

        if "pnl" in trades_with_equity.columns:
            trades_with_equity["cumulative_pnl"] = trades_with_equity["pnl"].cumsum()
            trades_with_equity["equity"] = (
                self.initial_capital + trades_with_equity["cumulative_pnl"]
            )
        else:
            # Si no hay pnl, asumir equity constante
            trades_with_equity["equity"] = self.initial_capital

        equity_curve = trades_with_equity[["timestamp", "equity"]].copy()

        return equity_curve

    def calculate_benchmark_returns(
        self, benchmark_prices: pd.DataFrame, price_col: str
    ) -> pd.DataFrame:
        """
        Calcula retornos del benchmark.

        Args:
            benchmark_prices: DataFrame con timestamp y precio
            price_col: Nombre de la columna de precio

        Returns:
            DataFrame con timestamp, returns
        """
        df = benchmark_prices.copy()
        df["returns"] = df[price_col].pct_change()
        df = df.dropna(subset=["returns"])

        return df[["timestamp", "returns"]]

    def calculate_roi_annualized(self, equity_curve: pd.DataFrame) -> float:
        """
        Calcula ROI anualizado del portfolio.

        Args:
            equity_curve: DataFrame con timestamp, equity

        Returns:
            ROI anualizado en porcentaje
        """
        if equity_curve.empty:
            return 0.0

        initial_equity = equity_curve.iloc[0]["equity"]
        final_equity = equity_curve.iloc[-1]["equity"]

        # Calcular días transcurridos
        days = (equity_curve.iloc[-1]["timestamp"] - equity_curve.iloc[0]["timestamp"]).days

        if days <= 0 or initial_equity <= 0:
            return 0.0

        # ROI total
        total_return = (final_equity - initial_equity) / initial_equity

        # Anualizar
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1

        return annualized_return * 100  # En porcentaje

    def calculate_alpha_beta(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> Tuple[float, float]:
        """
        Calcula Alpha y Beta vs benchmark.

        Alpha = Retorno del portfolio - (Risk-free rate + Beta * (Benchmark return - Risk-free rate))
        Beta = Cov(portfolio, benchmark) / Var(benchmark)

        Args:
            portfolio_returns: Serie de retornos del portfolio
            benchmark_returns: Serie de retornos del benchmark

        Returns:
            (alpha, beta)
        """
        # Alinear series
        aligned = pd.DataFrame(
            {"portfolio": portfolio_returns, "benchmark": benchmark_returns}
        ).dropna()

        if len(aligned) < 2:
            return 0.0, 1.0

        # Beta via regresión lineal
        slope, intercept, r_value, p_value, std_err = linregress(
            aligned["benchmark"], aligned["portfolio"]
        )
        beta = slope

        # Alpha = intercept anualizado
        # Convertir a términos anuales
        periods_per_year = 252  # Asumiendo retornos diarios
        alpha_annualized = intercept * periods_per_year * 100  # En porcentaje

        return alpha_annualized, beta

    def calculate_sharpe_rolling(
        self, equity_curve: pd.DataFrame, window: int = 30
    ) -> pd.DataFrame:
        """
        Calcula Sharpe Ratio rolling.

        Args:
            equity_curve: DataFrame con timestamp, equity
            window: Ventana en días

        Returns:
            DataFrame con timestamp, sharpe_ratio
        """
        df = equity_curve.copy()
        df["returns"] = df["equity"].pct_change()
        df = df.dropna(subset=["returns"])

        # Rolling Sharpe
        # Sharpe = (mean(returns) - risk_free_rate) / std(returns) * sqrt(periods_per_year)
        daily_rf = self.risk_free_rate / 252

        df["rolling_mean"] = df["returns"].rolling(window).mean()
        df["rolling_std"] = df["returns"].rolling(window).std()

        df["sharpe_ratio"] = (
            (df["rolling_mean"] - daily_rf) / df["rolling_std"]
        ) * np.sqrt(252)

        return df[["timestamp", "sharpe_ratio"]].dropna()

    def calculate_max_drawdown(self, equity_curve: pd.DataFrame) -> float:
        """
        Calcula Maximum Drawdown.

        Args:
            equity_curve: DataFrame con timestamp, equity

        Returns:
            Max Drawdown en porcentaje (valor negativo)
        """
        if equity_curve.empty:
            return 0.0

        equity = equity_curve["equity"].values
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max * 100

        return float(np.min(drawdown))

    def calculate_correlation(
        self, portfolio_returns: pd.Series, benchmark_returns: pd.Series
    ) -> float:
        """
        Calcula correlación entre portfolio y benchmark.

        Args:
            portfolio_returns: Serie de retornos del portfolio
            benchmark_returns: Serie de retornos del benchmark

        Returns:
            Coeficiente de correlación
        """
        aligned = pd.DataFrame(
            {"portfolio": portfolio_returns, "benchmark": benchmark_returns}
        ).dropna()

        if len(aligned) < 2:
            return 0.0

        return float(aligned.corr().iloc[0, 1])

    def generate_comparison_report(self, start_date: str, end_date: str) -> Dict:
        """
        Genera reporte completo de comparación con benchmarks.

        Args:
            start_date: Fecha inicial YYYY-MM-DD
            end_date: Fecha final YYYY-MM-DD

        Returns:
            Dict con métricas comparativas
        """
        # 1. Cargar trades y calcular equity curve
        self.load_trades()
        equity_curve = self.calculate_portfolio_equity_curve()

        # Filtrar equity curve por fechas
        equity_curve = equity_curve[
            (equity_curve["timestamp"] >= start_date) & (equity_curve["timestamp"] <= end_date)
        ]

        if equity_curve.empty:
            raise ValueError(f"No hay trades entre {start_date} y {end_date}")

        # 2. Descargar precios de benchmarks
        self.fetch_btc_prices(start_date, end_date)
        self.fetch_sp500_prices(start_date, end_date)

        # 3. Calcular métricas del portfolio
        roi_annualized = self.calculate_roi_annualized(equity_curve)
        max_dd = self.calculate_max_drawdown(equity_curve)
        sharpe_rolling = self.calculate_sharpe_rolling(equity_curve)

        # Sharpe promedio
        avg_sharpe = sharpe_rolling["sharpe_ratio"].mean() if not sharpe_rolling.empty else 0.0

        # Retornos del portfolio
        portfolio_returns = equity_curve["equity"].pct_change().dropna()

        # 4. Calcular métricas vs BTC
        btc_metrics = {}
        if not self.btc_prices.empty:
            btc_returns_df = self.calculate_benchmark_returns(self.btc_prices, "btc_price")

            # Alinear retornos
            merged_btc = pd.merge_asof(
                equity_curve.sort_values("timestamp"),
                btc_returns_df.sort_values("timestamp"),
                on="timestamp",
                direction="nearest",
            )
            btc_returns = merged_btc["returns"].dropna()

            alpha_btc, beta_btc = self.calculate_alpha_beta(portfolio_returns, btc_returns)
            corr_btc = self.calculate_correlation(portfolio_returns, btc_returns)

            # ROI de BTC
            btc_initial = self.btc_prices.iloc[0]["btc_price"]
            btc_final = self.btc_prices.iloc[-1]["btc_price"]
            btc_roi = (btc_final - btc_initial) / btc_initial * 100

            btc_metrics = {
                "roi_pct": float(btc_roi),
                "alpha": float(alpha_btc),
                "beta": float(beta_btc),
                "correlation": float(corr_btc),
            }

        # 5. Calcular métricas vs S&P500
        sp500_metrics = {}
        if not self.sp500_prices.empty:
            sp500_returns_df = self.calculate_benchmark_returns(self.sp500_prices, "sp500_price")

            merged_sp500 = pd.merge_asof(
                equity_curve.sort_values("timestamp"),
                sp500_returns_df.sort_values("timestamp"),
                on="timestamp",
                direction="nearest",
            )
            sp500_returns = merged_sp500["returns"].dropna()

            alpha_sp500, beta_sp500 = self.calculate_alpha_beta(portfolio_returns, sp500_returns)
            corr_sp500 = self.calculate_correlation(portfolio_returns, sp500_returns)

            # ROI de S&P500
            sp500_initial = self.sp500_prices.iloc[0]["sp500_price"]
            sp500_final = self.sp500_prices.iloc[-1]["sp500_price"]
            sp500_roi = (sp500_final - sp500_initial) / sp500_initial * 100

            sp500_metrics = {
                "roi_pct": float(sp500_roi),
                "alpha": float(alpha_sp500),
                "beta": float(beta_sp500),
                "correlation": float(corr_sp500),
            }

        # 6. Construir reporte
        report = {
            "generated_at": datetime.now().isoformat(),
            "period": {"start": start_date, "end": end_date},
            "portfolio": {
                "roi_annualized_pct": float(roi_annualized),
                "sharpe_ratio_avg": float(avg_sharpe),
                "max_drawdown_pct": float(max_dd),
                "initial_capital": float(self.initial_capital),
                "final_equity": float(equity_curve.iloc[-1]["equity"]),
            },
            "benchmarks": {
                "btc_usd": btc_metrics,
                "sp500": sp500_metrics,
            },
        }

        return report

    def save_report(self, report: Dict, output_path: str = str(BENCHMARK_COMPARISON_PATH)):
        """Guarda el reporte en JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"✅ Reporte guardado en: {output_path}")

    def plot_equity_curve(
        self,
        equity_curve: pd.DataFrame,
        btc_prices: Optional[pd.DataFrame] = None,
        sp500_prices: Optional[pd.DataFrame] = None,
        output_path: str = str(EQUITY_CURVE_PATH),
    ):
        """
        Genera gráfico comparativo de equity curves.

        Args:
            equity_curve: DataFrame del portfolio
            btc_prices: DataFrame de BTC (opcional)
            sp500_prices: DataFrame de S&P500 (opcional)
            output_path: Ruta de salida del gráfico
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        # Normalizar equity curve del portfolio
        portfolio_normalized = equity_curve.copy()
        portfolio_normalized["equity_norm"] = (
            portfolio_normalized["equity"] / portfolio_normalized.iloc[0]["equity"]
        ) * 100

        ax.plot(
            portfolio_normalized["timestamp"],
            portfolio_normalized["equity_norm"],
            label="Portfolio (Sistema Híbrido)",
            linewidth=2,
            color="blue",
        )

        # BTC
        if btc_prices is not None and not btc_prices.empty:
            btc_normalized = btc_prices.copy()
            btc_normalized["btc_norm"] = (
                btc_normalized["btc_price"] / btc_normalized.iloc[0]["btc_price"]
            ) * 100
            ax.plot(
                btc_normalized["timestamp"],
                btc_normalized["btc_norm"],
                label="BTC/USD",
                linewidth=2,
                color="orange",
                linestyle="--",
            )

        # S&P500
        if sp500_prices is not None and not sp500_prices.empty:
            sp500_normalized = sp500_prices.copy()
            sp500_normalized["sp500_norm"] = (
                sp500_normalized["sp500_price"] / sp500_normalized.iloc[0]["sp500_price"]
            ) * 100
            ax.plot(
                sp500_normalized["timestamp"],
                sp500_normalized["sp500_norm"],
                label="S&P500",
                linewidth=2,
                color="green",
                linestyle="-.",
            )

        ax.set_xlabel("Fecha", fontsize=12)
        ax.set_ylabel("Valor Normalizado (Base 100)", fontsize=12)
        ax.set_title("Equity Curve: Portfolio vs Benchmarks", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(output_path, dpi=300)
        print(f"✅ Equity curve guardada en: {output_path}")

        plt.close()


def main():
    """Función principal para ejecutar la evaluación de benchmark."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluación de benchmark del sistema")
    parser.add_argument(
        "--start",
        type=str,
        default=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
        help="Fecha inicial (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Fecha final (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--capital", type=float, default=100000.0, help="Capital inicial (default: 100000)"
    )

    args = parser.parse_args()

    evaluator = BenchmarkEvaluator(initial_capital=args.capital)

    print(f"🔍 Generando reporte de benchmark para {args.start} → {args.end}")

    # Generar reporte
    report = evaluator.generate_comparison_report(args.start, args.end)

    # Guardar reporte
    evaluator.save_report(report)

    # Generar equity curve
    equity_curve = evaluator.calculate_portfolio_equity_curve()
    equity_curve = equity_curve[
        (equity_curve["timestamp"] >= args.start) & (equity_curve["timestamp"] <= args.end)
    ]

    evaluator.plot_equity_curve(
        equity_curve, evaluator.btc_prices, evaluator.sp500_prices
    )

    # Mostrar resumen
    print("\n📊 RESUMEN DE BENCHMARK")
    print("=" * 60)
    print(f"Portfolio ROI (anualizado): {report['portfolio']['roi_annualized_pct']:.2f}%")
    print(f"Sharpe Ratio (promedio):    {report['portfolio']['sharpe_ratio_avg']:.2f}")
    print(f"Max Drawdown:               {report['portfolio']['max_drawdown_pct']:.2f}%")
    print("\n📈 vs BTC/USD:")
    if report["benchmarks"]["btc_usd"]:
        print(f"  BTC ROI:        {report['benchmarks']['btc_usd']['roi_pct']:.2f}%")
        print(f"  Alpha:          {report['benchmarks']['btc_usd']['alpha']:.2f}%")
        print(f"  Beta:           {report['benchmarks']['btc_usd']['beta']:.2f}")
        print(f"  Correlación:    {report['benchmarks']['btc_usd']['correlation']:.2f}")

    print("\n📊 vs S&P500:")
    if report["benchmarks"]["sp500"]:
        print(f"  S&P500 ROI:     {report['benchmarks']['sp500']['roi_pct']:.2f}%")
        print(f"  Alpha:          {report['benchmarks']['sp500']['alpha']:.2f}%")
        print(f"  Beta:           {report['benchmarks']['sp500']['beta']:.2f}")
        print(f"  Correlación:    {report['benchmarks']['sp500']['correlation']:.2f}")

    print("\n✅ Evaluación completada.")


if __name__ == "__main__":
    main()
