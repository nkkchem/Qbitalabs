#!/bin/bash
# QBitaLabs Training Script Runner
# Wrapper for MVP model training

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$PROJECT_ROOT/mvp/logs"
VENV_DIR="$PROJECT_ROOT/.venv"

mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_$TIMESTAMP.log"

echo "============================================" | tee -a "$LOG_FILE"
echo "QBitaLabs MVP Training" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

# Activate virtual environment
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    echo "Virtual environment activated" | tee -a "$LOG_FILE"
fi

cd "$PROJECT_ROOT"

# Default config
CONFIG="configs/training/m4_mac.yaml"
MODELS="all"
QUICK=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --quick)
            QUICK="--quick"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "Config: $CONFIG" | tee -a "$LOG_FILE"
echo "Models: $MODELS" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run training
python scripts/train/train_mvp.py \
    --config "$CONFIG" \
    --models $MODELS \
    --data-dir "./mvp/data" \
    --output-dir "./mvp" \
    $QUICK 2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"
echo "Training completed (exit code: $EXIT_CODE)" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "============================================" | tee -a "$LOG_FILE"

exit $EXIT_CODE
