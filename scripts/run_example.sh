#!/bin/bash
set -euo pipefail

# Basic configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/path/to/your/model}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${PROJECT_DIR}/outputs}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_DIR}/data}"
SUBMIT_INTERVAL="${SUBMIT_INTERVAL:-1}"

# Task configuration
declare -A TASKS=(
    ["medqa"]="test_short_medqa.json"
)

# Common parameters
COMMON_PARAMS=(
    "--model" "$MODEL_PATH"
    "--temperature" "0.6"
    "--top_p" "0.95"
    "--top_k" "20"
    "--max_tokens" "2048"
    "--num_generations" "4"
    "--question_type" "mcq"
)

# Launch all tasks
for task_name in "${!TASKS[@]}"; do
    dataset_file="${TASKS[$task_name]}"
    output_prefix="demo-${task_name}"

    echo "========================================"
    echo "Running task: $task_name"
    echo "Model path: $MODEL_PATH"
    echo "Dataset: ${DATASET_DIR}/${dataset_file}"

    python "${PROJECT_DIR}/vllm_logitsbias_multi.py" \
        "${COMMON_PARAMS[@]}" \
        --dataset "${DATASET_DIR}/${dataset_file}" \
        --output_prefix "${BASE_OUTPUT_DIR}/${output_prefix}"

    sleep "$SUBMIT_INTERVAL"
done
