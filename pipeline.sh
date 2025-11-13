#!/usr/bin/env bash
set -e

echo "[1/3] Running Clap Analysis..."
python3 src/main.py       # writes data/output.json

echo "[2/3] Compiling Visualization Program..."
make -C visuals

echo "[3/3] Running Visualization Program..."
./visuals/drones