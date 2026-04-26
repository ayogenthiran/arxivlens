#!/bin/bash

echo "Setting up ArxivLens environment..."

# Create necessary directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/vector_store

# Optionally create a .env if it doesn't exist
if [ ! -f .env ]; then
  echo "No .env file found. Creating a template .env"
  cat > .env << 'EOF'
HOST=127.0.0.1
PORT=8000
DATA_DIR=data/raw
OUTPUT_DIR=data/processed
VECTOR_STORE_DIR=data/vector_store
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
OPENAI_API_KEY=
OPENAI_API_BASE=https://api.openai.com/v1
EOF
else
  echo ".env file already exists."
fi

echo "Setup complete. ✅"
echo ""
echo "Next steps:"
echo "1) pip install -r requirements.txt"
echo "2) python main.py --process-data   # one-time"
echo "3) python main.py                  # backend"
echo "4) streamlit run streamlit_app.py  # frontend"
