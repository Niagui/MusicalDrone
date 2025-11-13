#!/usr/bin/env bash
set -e

echo "[1/3] Running Python..."
python3 src/main.py       # writes data/output.json

echo "[2/3] Running make..."
make -C visuals

echo "[3/3] Running C++ program..."
./visuals/drones 