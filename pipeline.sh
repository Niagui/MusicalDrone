#!/usr/bin/env bash

set -euo pipefail

OUTPUT_CSV="trajectories.csv"
AUDIO_INPUT="testSong.mp3"
CACHE_ROOT="data"
USE_LLM=0
RUN_EVALUATION=1
DESCRIPTOR_ANCHORS=0
ANCHOR_CONFIG="json/descriptor_anchor_config.json"
USE_PHRASE_PLAN=0

usage() {
    echo "Usage: $0 [--output output.csv] [--audio song.mp3] [--cache-root data] [--llm] [--descriptor-anchors] [--anchor-config config.json] [--phrase-plan] [--no-phrase-plan] [--no-eval]"
}

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
            USE_PHRASE_PLAN=1
            shift
            ;;
        --descriptor-anchors)
            DESCRIPTOR_ANCHORS=1
            shift
            ;;
        --anchor-config)
            ANCHOR_CONFIG="$2"
            shift 2
            ;;
        --phrase-plan)
            USE_PHRASE_PLAN=1
            shift
            ;;
        --no-phrase-plan)
            USE_PHRASE_PLAN=0
            shift
            ;;
        --no-eval|--no-evaluation)
            RUN_EVALUATION=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            usage
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

echo "[data] Writing audio artifacts to $CACHE_DIR"

if [[ "$DESCRIPTOR_ANCHORS" -eq 1 ]]; then
    if [[ "$USE_PHRASE_PLAN" -eq 1 ]]; then
        if [[ "$RUN_EVALUATION" -eq 1 ]]; then
            TOTAL_STEPS=6
        else
            TOTAL_STEPS=5
        fi
        echo "[1/$TOTAL_STEPS] Preparing generated analysis JSON..."
        DRONE_JSON_DIR="$JSON_CACHE_DIR" python3 src/main.py --audio "$AUDIO_INPUT" --prepare-only

        echo "[2/$TOTAL_STEPS] Generating phrase plan..."
        DRONE_JSON_DIR="$JSON_CACHE_DIR" python3 src/phrase_generator.py

        echo "[3/$TOTAL_STEPS] Building descriptor-anchor CLAP weights..."
        DRONE_JSON_DIR="$JSON_CACHE_DIR" python3 src/main.py \
            --audio "$AUDIO_INPUT" \
            --descriptor-anchors \
            --anchor-config "$ANCHOR_CONFIG"

        COMPILE_STEP=4
        RUN_STEP=5
        EVALUATION_STEP=6
    else
        if [[ "$RUN_EVALUATION" -eq 1 ]]; then
            TOTAL_STEPS=4
        else
            TOTAL_STEPS=3
        fi
        echo "[1/$TOTAL_STEPS] Running descriptor-anchor audio analysis..."
        DRONE_JSON_DIR="$JSON_CACHE_DIR" python3 src/main.py \
            --audio "$AUDIO_INPUT" \
            --descriptor-anchors \
            --anchor-config "$ANCHOR_CONFIG"

        COMPILE_STEP=2
        RUN_STEP=3
        EVALUATION_STEP=4
    fi
elif [[ "$USE_LLM" -eq 1 ]]; then
    if [[ "$RUN_EVALUATION" -eq 1 ]]; then
        if [[ "$USE_PHRASE_PLAN" -eq 1 ]]; then
            TOTAL_STEPS=7
        else
            TOTAL_STEPS=6
        fi
    else
        if [[ "$USE_PHRASE_PLAN" -eq 1 ]]; then
            TOTAL_STEPS=6
        else
            TOTAL_STEPS=5
        fi
    fi
    echo "[1/$TOTAL_STEPS] Preparing generated analysis JSON..."
    DRONE_JSON_DIR="$JSON_CACHE_DIR" python3 src/main.py --audio "$AUDIO_INPUT" --prepare-only

    echo "[2/$TOTAL_STEPS] Generating LLM label variations..."
    DRONE_JSON_DIR="$JSON_CACHE_DIR" python3 src/label_variations_generator.py

    if [[ "$USE_PHRASE_PLAN" -eq 1 ]]; then
        echo "[3/$TOTAL_STEPS] Generating phrase plan..."
        DRONE_JSON_DIR="$JSON_CACHE_DIR" python3 src/phrase_generator.py

        echo "[4/$TOTAL_STEPS] Building CLAP weights with generated LLM outputs..."
        DRONE_JSON_DIR="$JSON_CACHE_DIR" python3 src/main.py --audio "$AUDIO_INPUT" --use-llm

        COMPILE_STEP=5
        RUN_STEP=6
        EVALUATION_STEP=7
    else
        echo "[3/$TOTAL_STEPS] Building CLAP weights with generated LLM outputs..."
        DRONE_JSON_DIR="$JSON_CACHE_DIR" python3 src/main.py --audio "$AUDIO_INPUT" --use-llm

        COMPILE_STEP=4
        RUN_STEP=5
        EVALUATION_STEP=6
    fi
else
    if [[ "$RUN_EVALUATION" -eq 1 ]]; then
        TOTAL_STEPS=4
    else
        TOTAL_STEPS=3
    fi
    echo "[1/$TOTAL_STEPS] Running audio analysis..."
    DRONE_JSON_DIR="$JSON_CACHE_DIR" python3 src/main.py --audio "$AUDIO_INPUT"

    COMPILE_STEP=2
    RUN_STEP=3
    EVALUATION_STEP=4
fi

echo "[$COMPILE_STEP/$TOTAL_STEPS] Compiling Waypoint Planner Program..."
make trajectories -C visuals

echo "[$RUN_STEP/$TOTAL_STEPS] Running Waypoint Planner Program..."
DRONE_JSON_DIR="$JSON_CACHE_DIR" DRONE_USE_PHRASE_PLAN="$USE_PHRASE_PLAN" ./visuals/traj > "$CACHED_TRAJECTORY"

if [[ "$OUTPUT_CSV" != "$CACHED_TRAJECTORY" ]]; then
    cp "$CACHED_TRAJECTORY" "$OUTPUT_CSV"
fi

if [[ "$RUN_EVALUATION" -eq 1 ]]; then
    echo "[$EVALUATION_STEP/$TOTAL_STEPS] Evaluating generated trajectory..."
    python3 evaluate.py "$CACHE_DIR"
fi

echo "[done] JSON dir: $JSON_CACHE_DIR"
echo "[done] Trajectory: $CACHED_TRAJECTORY"
if [[ "$RUN_EVALUATION" -eq 1 ]]; then
    echo "[done] Evaluation dir: $CACHE_DIR/evaluation"
fi
if [[ "$OUTPUT_CSV" != "$CACHED_TRAJECTORY" ]]; then
    echo "[done] Output copy: $OUTPUT_CSV"
fi
