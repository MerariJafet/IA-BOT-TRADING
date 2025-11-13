# Arquitectura del sistema IA BOT TRADING

## Módulos principales
- **core/**: manejo de datos, tokenización, secuencias, logging, contratos.
- **models/**: embeddings, redes neuronales, política de acción.
- **backtest/**: simulación, scoring, monetización.
- **utils/**: funciones auxiliares y validaciones.

## Flujo lógico
1. Datos → Tokenización → Secuencias
2. Secuencias → Embeddings (espacio HD)
3. Embeddings → Descubrimiento de patrones
4. Patrones → Política de acción → Backtest
5. Validación y fine-tuning continuo.

## KPI Global
Profit Factor ≥ 0.8 | Sharpe ≥ 0.5 | MaxDD ≤ 10%

## Infraestructura de despliegue
- **Contenerización**: Construida sobre Docker (`Dockerfile`) y orquestada con `docker-compose` para bot principal, scheduler y monitor.
- **CI/CD**: GitHub Actions (`.github/workflows/deploy.yml`) prepara imagen y push a registro (pendiente integración ECR/GCR).

## Auto-escalado y resiliencia
- **Estrategia**: Despliegue objetivo en AWS ECS Fargate o GCP Cloud Run con reglas horizontales.
- **Escala Out**: CPU promedio > 50% durante 5 minutos -> adicionar tarea/instancia.
- **Escala In**: CPU < 10% sostenida 5 minutos -> remover tarea/instancia sin afectar disponibilidad.
- **Healthcheck**: Endpoint lógico vía `src/utils/healthcheck.py` para sondas de plataforma.

## Monitoreo y alertas
- **Disponibilidad**: Integrar UptimeRobot y Healthchecks.io para chequeos HTTP periódicos.
- **Logs**: Centralizar en CloudWatch (AWS) o Cloud Logging (GCP) para correlación y alertas.
- **Alertas Telegram**: `src/utils/telegram_alerts.py` envía notificaciones al canal operativo.
