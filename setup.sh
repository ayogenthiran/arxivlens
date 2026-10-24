#!/bin/bash

echo "Setting up ArxivLens environment..."

# Create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/vector_store

# Optionally create a .env if it doesn't exist
if [ ! -f .env ]; then
  echo "No .env file found. Copying .env.example"
  cp .env.example .env
else
  echo ".env file already exists."
fi

echo "Setup complete. ✅"
echo ""
echo "Next steps:"
echo "1) pip install -r requirements.txt   # or: uv sync --frozen"
echo "2) python main.py --process-data   # one-time"
echo "3) python main.py                  # backend"
echo "4) streamlit run streamlit_app.py  # frontend"
