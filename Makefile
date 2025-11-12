PYTHON := .venv/bin/python

setup:
	pip install -r requirements.txt
	touch .env
	echo 'Setup complete.'

test:
	pytest -q

lint:
	flake8 src/ --max-line-length=100
	isort . --check-only --profile black
	black --check .

format:
	isort . --profile black
	black .

run:
	$(PYTHON) src/main.py || echo 'main.py aún no implementado'

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
