#!/bin/zsh

# Run etl, train and predict model in one flow.
python3 etl.py
echo "[1/3] Extracting, transforming and loading raw data..."

python3 train.py
echo "[2/3] Training model..."

python3 predict.py
echo "[3/3] Generating predictions..."

echo "[DONE] Task completed."
