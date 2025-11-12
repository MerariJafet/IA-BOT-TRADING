# IA BOT TRADING

Sistema de inteligencia artificial para trading basado en análisis de secuencias de tokens, espacios de alta dimensión y redes neuronales adaptativas.

## Estructura general

```
IA BOT TRADING/
├── data/              # datasets crudos y procesados
├── notebooks/          # experimentos y pruebas exploratorias
├── src/                # código fuente principal
│   ├── core/           # manejo de datos, tokenización y secuencias
│   ├── models/         # arquitecturas neuronales, embeddings y políticas
│   ├── utils/          # herramientas auxiliares
│   └── backtest/       # simulación, scoring y validación económica
├── reports/            # resultados y métricas de monetización
├── configs/            # parámetros de configuración
├── docs/               # documentación extendida
└── tests/              # pruebas unitarias
```

## KPI del sistema
- Profit Factor ≥ 0.8
- Sharpe Ratio ≥ 0.5
- Max Drawdown ≤ 10%

## Sprint 0 - Objetivo
Configurar entorno técnico y base de desarrollo reproducible.

## Autenticación opcional con Binance
Para ejecutar endpoints que requieran firma o límites extendidos:
- Define `BINANCE_API_KEY` y `BINANCE_API_SECRET` en el archivo `.env` en la raíz del proyecto (no se versiona).
- Ejecuta `make check_env` para validar que las variables se cargan correctamente y que la API responde usando la API Key configurada.
- El comando imprime confirmación de las credenciales y realiza una llamada autenticada a `exchangeInfo`. Úsalo antes de correr pipelines que dependan de acceso autenticado.

## Tokenización dinámica (Dollar & Imbalance bars)
- Coloca los históricos normalizados de Binance en `data/historical_1y_parquet/<SIMBOLO>.parquet`.
- Corre `make tokenize` y completa los prompts (símbolo, método y threshold) para generar secuencias en `data/tokens/`.
- Los métodos disponibles son `dollar` (acumulación por volumen nocional) e `imbalance` (desequilibrio compra/venta). Ambos generan archivos Parquet que sirven como entrada al módulo de embeddings.
