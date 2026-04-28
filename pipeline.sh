#!/usr/bin/env bash

set -euo pipefail

OUTPUT_CSV="trajectories.csv"
AUDIO_INPUT="testSong.mp3"
CACHE_ROOT="cache"
USE_LLM=0
RUN_EVALUATION=1

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
        --no-eval|--no-evaluation)
            RUN_EVALUATION=0
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--output output.csv] [--audio song.mp3] [--cache-root cache] [--llm] [--no-eval]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--output output.csv] [--audio song.mp3] [--cache-root cache] [--llm] [--no-eval]"
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
    if [[ "$RUN_EVALUATION" -eq 1 ]]; then
        TOTAL_STEPS=7
    else
        TOTAL_STEPS=6
    fi
    echo "[1/$TOTAL_STEPS] Preparing cached analysis JSON..."
    DRONE_JSON_DIR="$JSON_CACHE_DIR" python3 src/main.py --audio "$AUDIO_INPUT" --prepare-only

    echo "[2/$TOTAL_STEPS] Generating cached LLM label variations..."
    DRONE_JSON_DIR="$JSON_CACHE_DIR" python3 src/label_variations_generator.py

    echo "[3/$TOTAL_STEPS] Generating cached phrase plan..."
    DRONE_JSON_DIR="$JSON_CACHE_DIR" python3 src/phrase_generator.py

    echo "[4/$TOTAL_STEPS] Building CLAP weights with cached LLM outputs..."
    DRONE_JSON_DIR="$JSON_CACHE_DIR" python3 src/main.py --audio "$AUDIO_INPUT" --use-llm
else
    if [[ "$RUN_EVALUATION" -eq 1 ]]; then
        TOTAL_STEPS=4
    else
        TOTAL_STEPS=3
    fi
    echo "[1/$TOTAL_STEPS] Running cached audio analysis..."
    DRONE_JSON_DIR="$JSON_CACHE_DIR" python3 src/main.py --audio "$AUDIO_INPUT"
fi

if [[ "$USE_LLM" -eq 1 ]]; then
    echo "[5/$TOTAL_STEPS] Compiling Waypoint Planner Program..."
else
    echo "[2/$TOTAL_STEPS] Compiling Waypoint Planner Program..."
fi
make trajectories -C visuals

if [[ "$USE_LLM" -eq 1 ]]; then
    echo "[6/$TOTAL_STEPS] Running Waypoint Planner Program..."
else
    echo "[3/$TOTAL_STEPS] Running Waypoint Planner Program..."
fi
DRONE_JSON_DIR="$JSON_CACHE_DIR" ./visuals/traj > "$CACHED_TRAJECTORY"

if [[ "$OUTPUT_CSV" != "$CACHED_TRAJECTORY" ]]; then
    cp "$CACHED_TRAJECTORY" "$OUTPUT_CSV"
fi

if [[ "$RUN_EVALUATION" -eq 1 ]]; then
    if [[ "$USE_LLM" -eq 1 ]]; then
        echo "[7/$TOTAL_STEPS] Evaluating generated trajectory..."
    else
        echo "[4/$TOTAL_STEPS] Evaluating generated trajectory..."
    fi
    python3 evaluate.py "$CACHE_DIR"
fi

echo "[done] Cached JSON dir: $JSON_CACHE_DIR"
echo "[done] Cached trajectory: $CACHED_TRAJECTORY"
if [[ "$RUN_EVALUATION" -eq 1 ]]; then
    echo "[done] Evaluation dir: $CACHE_DIR/evaluation"
fi
if [[ "$OUTPUT_CSV" != "$CACHED_TRAJECTORY" ]]; then
    echo "[done] Output copy: $OUTPUT_CSV"
fi
