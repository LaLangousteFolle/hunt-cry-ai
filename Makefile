"""Makefile for Hunt Showdown Sound project.

Usage:
    make install    - Install dependencies
    make train      - Train the model
    make predict    - Run prediction on test file
    make test       - Run unit tests
    make lint       - Check code style
    make clean      - Remove generated files
"""

.PHONY: install train predict test lint clean help

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

install-dev:
	@echo "Installing development dependencies..."
	pip install -r requirements.txt black flake8 pytest pytest-cov

train:
	@echo "Training model..."
	python -m src.train

predict:
	@echo "Running prediction..."
	python -m src.predict

test:
	@echo "Running unit tests..."
	pytest tests/ -v --cov=src

lint:
	@echo "Checking code style..."
	black --check src/ tests/
	flake8 src/ tests/ --max-line-length=120

format:
	@echo "Formatting code..."
	black src/ tests/

clean:
	@echo "Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned up!"

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make install-dev   - Install dev dependencies"
	@echo "  make train         - Train the model"
	@echo "  make predict       - Run prediction"
	@echo "  make test          - Run unit tests"
	@echo "  make lint          - Check code style"
	@echo "  make format        - Format code with black"
	@echo "  make clean         - Remove generated files"
	@echo "  make help          - Show this help"
