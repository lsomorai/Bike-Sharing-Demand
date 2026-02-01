.PHONY: install test lint clean notebook

install:
	pip install -r requirements.txt

notebook:
	jupyter notebook 01_eda_and_modeling.ipynb

lint:
	python -m py_compile src/evaluate.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

help:
	@echo "Available commands:"
	@echo "  make install   - Install dependencies"
	@echo "  make notebook  - Launch Jupyter notebook"
	@echo "  make lint      - Check Python syntax"
	@echo "  make clean     - Remove cache files"
