PYTHON := .venv/bin/python

setup:
	pip install -r requirements.txt
	touch .env
	echo 'Setup complete.'

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m flake8 src/ --max-line-length=100
	$(PYTHON) -m isort . --check-only --profile black
	$(PYTHON) -m black --check .

format:
	$(PYTHON) -m isort . --profile black
	$(PYTHON) -m black .

run:
	$(PYTHON) src/main.py || echo 'main.py aún no implementado'

check_env:
	@echo 'Verificando variables de entorno...'
	@$(PYTHON) -c "from src.utils.env_loader import get_binance_credentials; api_key, api_secret = get_binance_credentials(); print('✔ API Key encontrada'); print('✔ API Secret encontrada')"
	@$(PYTHON) -m src.core.binance_test_connection

tokenize:
	@read -p 'Símbolo (ej. BTCUSDT): ' symbol; \
	read -p 'Método [dollar/imbalance]: ' method; \
	read -p 'Threshold (ej. 10000): ' threshold; \
	SYMBOL="$$symbol" METHOD="$$method" THRESHOLD="$$threshold" $(PYTHON) -c "import os; from src.core.tokenizer import tokenize_symbol; symbol = os.environ.get('SYMBOL') or 'BTCUSDT'; method = (os.environ.get('METHOD') or 'dollar').strip().lower(); threshold_env = os.environ.get('THRESHOLD') or '10000'; threshold = float(threshold_env); tokenize_symbol(symbol=symbol, method=method, threshold=threshold)"

sequences:
	@echo 'Generando secuencias multi-escala...'
	@$(PYTHON) -c "from src.core.sequence_builder import generate_sequences; symbol='BTCUSDT'; print(f'✅ Iniciando generación de secuencias para {symbol}'); generate_sequences(symbol)"

embeddings:
	@echo 'Construyendo embeddings PCA...'
	@PYTHONPATH="$(PWD)" $(PYTHON) -m src.core.feature_embedding

clusters:
	@echo 'Ejecutando clustering de patrones...'
	@PYTHONPATH="$(PWD)" $(PYTHON) -m src.core.pattern_clustering

freeze:
	pip freeze > requirements_locked.txt

help:
	@echo 'Comandos disponibles:'
	@echo '  make setup     -> Instala dependencias y prepara entorno'
	@echo '  make test      -> Ejecuta pruebas unitarias'
	@echo '  make lint      -> Verifica estilo de código'
	@echo '  make format    -> Aplica formateo automático'
	@echo '  make run       -> Ejecuta el bot principal'
	@echo '  make freeze    -> Actualiza requirements_locked.txt'
