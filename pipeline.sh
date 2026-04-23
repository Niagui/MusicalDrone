#!/usr/bin/env bash

set -euo pipefail

OUTPUT_CSV="trajectories.csv"
AUDIO_INPUT="testSong.mp3"
CACHE_ROOT="cache"
USE_LLM=0

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
        --cache-root)
            CACHE_ROOT="$2"
            shift 2
            ;;
        --llm)
            USE_LLM=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--output output.csv] [--audio song.mp3] [--cache-root cache] [--llm]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--output output.csv] [--audio song.mp3] [--cache-root cache] [--llm]"
            exit 1
            ;;
    esac
done

AUDIO_BASENAME="$(basename "$AUDIO_INPUT")"
CACHE_DIR="$CACHE_ROOT/$AUDIO_BASENAME"
JSON_CACHE_DIR="$CACHE_DIR/json"
CACHED_TRAJECTORY="$CACHE_DIR/trajectory.csv"

mkdir -p "$JSON_CACHE_DIR"

OUTPUT_DIR="$(dirname "$OUTPUT_CSV")"
mkdir -p "$OUTPUT_DIR"

echo "[cache] Writing audio artifacts to $CACHE_DIR"

if [[ "$USE_LLM" -eq 1 ]]; then
    echo "[1/6] Preparing cached analysis JSON..."
    DRONE_JSON_DIR="$JSON_CACHE_DIR" python3 src/main.py --audio "$AUDIO_INPUT" --prepare-only

    echo "[2/6] Generating cached LLM label variations..."
    DRONE_JSON_DIR="$JSON_CACHE_DIR" python3 src/label_variations_generator.py

    echo "[3/6] Generating cached phrase plan..."
    DRONE_JSON_DIR="$JSON_CACHE_DIR" python3 src/phrase_generator.py

    echo "[4/6] Building CLAP weights with cached LLM outputs..."
    DRONE_JSON_DIR="$JSON_CACHE_DIR" python3 src/main.py --audio "$AUDIO_INPUT" --use-llm
else
    echo "[1/3] Running cached audio analysis..."
    DRONE_JSON_DIR="$JSON_CACHE_DIR" python3 src/main.py --audio "$AUDIO_INPUT"
fi

if [[ "$USE_LLM" -eq 1 ]]; then
    echo "[5/6] Compiling Waypoint Planner Program..."
else
    echo "[2/3] Compiling Waypoint Planner Program..."
fi
make trajectories -C visuals

if [[ "$USE_LLM" -eq 1 ]]; then
    echo "[6/6] Running Waypoint Planner Program..."
else
    echo "[3/3] Running Waypoint Planner Program..."
fi
DRONE_JSON_DIR="$JSON_CACHE_DIR" ./visuals/traj > "$CACHED_TRAJECTORY"

if [[ "$OUTPUT_CSV" != "$CACHED_TRAJECTORY" ]]; then
    cp "$CACHED_TRAJECTORY" "$OUTPUT_CSV"
fi

echo "[done] Cached JSON dir: $JSON_CACHE_DIR"
echo "[done] Cached trajectory: $CACHED_TRAJECTORY"
if [[ "$OUTPUT_CSV" != "$CACHED_TRAJECTORY" ]]; then
    echo "[done] Output copy: $OUTPUT_CSV"
fi
