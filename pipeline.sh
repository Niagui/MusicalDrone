#!/usr/bin/env bash

set -e

OUTPUT_CSV="trajectories.csv"
AUDIO_INPUT="testSong.mp3"
USE_HIERARCHICAL=0
HIERARCHICAL_MACRO_PLANNER=""

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
        --hierarchical)
            USE_HIERARCHICAL=1
            shift
            ;;
        --hierarchical-macro-planner)
            HIERARCHICAL_MACRO_PLANNER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--output output.csv] [--audio song.mp3] [--hierarchical] [--hierarchical-macro-planner llm|neutral|auto]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--output output.csv] [--audio song.mp3] [--hierarchical] [--hierarchical-macro-planner llm|neutral|auto]"
            exit 1
            ;;
    esac
done

echo "[1/3] Running Clap Analysis..."
if [[ "$USE_HIERARCHICAL" -eq 1 ]]; then
    if [[ -n "$HIERARCHICAL_MACRO_PLANNER" ]]; then
        python3 src/main.py --audio "$AUDIO_INPUT" --hierarchical --hierarchical-macro-planner "$HIERARCHICAL_MACRO_PLANNER"
    else
        python3 src/main.py --audio "$AUDIO_INPUT" --hierarchical
    fi
else
    python3 src/main.py --audio "$AUDIO_INPUT"
fi

echo "[2/3] Compiling Waypoint Planner Program..."
make trajectories -C visuals

echo "[3/3] Running Waypoint Planner Program..."
if [[ "$USE_HIERARCHICAL" -eq 1 ]]; then
    DRONE_USE_HIERARCHICAL=1 ./visuals/traj > "$OUTPUT_CSV"
else
    ./visuals/traj > "$OUTPUT_CSV"
fi
