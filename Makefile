# Regras
.PHONY: help install run clean

help: ## Mostra os comandos disponíveis
	@echo "Comandos disponíveis:"
	@echo "  make install  - Instala as dependências do projeto"
	@echo "  make run      - Executa o script water_quality.py"
	@echo "  make clean    - Remove arquivos temporários"


clean: ## Remove arquivos temporários
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +

lock: ## Lock das dependencias
	@python3 -m pip install -q poetry==2.0.1
	@poetry lock

install: ## Instala as dependências do projeto
	@poetry install --no-root

run: ## Executa o script water_quality.py os modulos necessários são instalados automaticamente
	@poetry run python3 water_quality.py
