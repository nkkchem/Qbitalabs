#!/bin/bash
# QBitaLabs Data Download Script
# Downloads all required datasets for MVP training

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
DATA_DIR="$PROJECT_ROOT/mvp/data"
VENV_DIR="$PROJECT_ROOT/.venv"

echo "============================================"
echo "QBitaLabs Data Download"
echo "============================================"
echo ""

# Create data directory
mkdir -p "$DATA_DIR"

# Activate virtual environment
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    echo "Virtual environment activated"
fi

cd "$PROJECT_ROOT"

# Parse arguments
INCREMENTAL=""
DATASETS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --incremental)
            INCREMENTAL="--incremental"
            shift
            ;;
        --datasets)
            DATASETS="--datasets $2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Run download script
python scripts/data/download_all.py \
    --output-dir "$DATA_DIR" \
    $INCREMENTAL \
    $DATASETS

echo ""
echo "============================================"
echo "Download complete!"
echo "Data directory: $DATA_DIR"
echo "============================================"
