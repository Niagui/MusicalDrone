#!/usr/bin/env bash

set -e

OUTPUT_CSV="trajectories.csv"
AUDIO_INPUT="testSong.mp3"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            OUTPUT_CSV="$2"
            shift 2
            ;;
        --audio)
            AUDIO_INPUT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--output output.csv] [--audio song.mp3]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--output output.csv] [--audio song.mp3]"
            exit 1
            ;;
    esac
done

echo "[1/3] Running Clap Analysis..."
python3 src/main.py --audio "$AUDIO_INPUT"    # writes data/output.json

echo "[2/3] Compiling Waypoint Planner Program..."
make trajectories -C visuals

echo "[3/3] Running Waypoint Plannern Program..."
./visuals/traj > "$OUTPUT_CSV"
