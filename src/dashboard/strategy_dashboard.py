"""
Strategy Dashboard - Dashboard interactivo para visualización de estrategias.

Dashboard Streamlit que muestra señales de trading en vivo, métricas de rendimiento
y visualizaciones interactivas del desempeño de la estrategia.
"""

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.core.logger import get_logger

logger = get_logger(__name__)

# Configuración de la página
st.set_page_config(
    page_title="Trading Strategy Dashboard", layout="wide", page_icon="📈"
)

st.title("📈 Live Strategy Dashboard")
st.markdown("---")


def load_simulation_metrics():
    """Carga métricas de simulación si existen."""
    metrics_path = Path("reports/simulation_metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return None


def load_pattern_strengths():
    """Carga información de pattern strengths si existe."""
    patterns_path = Path("data/reinforcement/pattern_strengths.parquet")
    if patterns_path.exists():
        return pd.read_parquet(patterns_path)
    return None


# Sidebar con información del sistema
with st.sidebar:
    st.header("⚙️ Sistema")

    # Métricas de simulación
    metrics = load_simulation_metrics()
    if metrics:
        st.subheader("📊 Métricas de Simulación")
        st.metric("PnL Total", f"${metrics.get('pnl_total', 0):.2f}")
        st.metric("Sharpe Ratio", f"{metrics.get('sharpe', 0):.3f}")
        st.metric("Max Drawdown", f"${metrics.get('max_drawdown', 0):.2f}")
        st.metric("Num Trades", metrics.get("num_trades", 0))
    else:
        st.info("No hay métricas de simulación disponibles")

    # Información de patrones
    patterns = load_pattern_strengths()
    if patterns is not None and "strength" in patterns.columns:
        st.subheader("🧩 Patrones")
        st.metric("Total Patrones", len(patterns))
        st.metric("Fuerza Promedio", f"{patterns['strength'].mean():.3f}")
        st.metric("Fuerza Máxima", f"{patterns['strength'].max():.3f}")
    else:
        st.info("No hay patrones disponibles")

    st.markdown("---")
    st.caption("IA BOT TRADING v1.8")


# Panel principal
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Estado de Señales")

    # Placeholder para señales en vivo
    signal_placeholder = st.empty()

    # Mostrar estado actual
    signal_placeholder.info("Esperando señales en vivo...")

with col2:
    st.subheader("📈 Rendimiento")

    # Placeholder para métricas en tiempo real
    perf_placeholder = st.empty()

    if metrics:
        perf_placeholder.success(f"PnL: ${metrics.get('pnl_total', 0):.2f}")
    else:
        perf_placeholder.warning("Ejecuta una simulación primero")


# Sección de visualización de datos
st.markdown("---")
st.subheader("📊 Visualización de Datos")

tab1, tab2, tab3 = st.tabs(["Patrones", "Secuencias", "Simulación"])

with tab1:
    patterns = load_pattern_strengths()
    if patterns is not None:
        st.dataframe(patterns.head(20), use_container_width=True)

        if "strength" in patterns.columns:
            st.bar_chart(patterns["strength"].head(20))
    else:
        st.info("No hay datos de patrones. Ejecuta `make patterns` primero.")

with tab2:
    seq_path = Path("data/sequences")
    if seq_path.exists():
        seq_files = list(seq_path.glob("*.parquet"))
        if seq_files:
            st.write(f"Archivos de secuencia encontrados: {len(seq_files)}")
            for f in seq_files[:5]:
                st.text(f"• {f.name}")
        else:
            st.info("No hay archivos de secuencias.")
    else:
        st.info("No hay directorio de secuencias. Ejecuta `make sequences` primero.")

with tab3:
    if metrics:
        st.json(metrics)
    else:
        st.info("No hay métricas de simulación. Ejecuta `make simulate` primero.")


# Footer
st.markdown("---")
st.caption(
    "💡 Tip: Usa `make live` para iniciar el emulador de señales en tiempo real"
)


# Modo de actualización automática
if st.checkbox("🔄 Auto-refresh (cada 5s)"):
    time.sleep(5)
    st.rerun()
