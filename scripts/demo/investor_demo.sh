#!/bin/bash
# QBitaLabs Investor Demo Runner
# Interactive demonstration for investors

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
VENV_DIR="$PROJECT_ROOT/.venv"

echo ""
echo "  ╔═══════════════════════════════════════════════════════╗"
echo "  ║                                                       ║"
echo "  ║              Q B I T A L A B S                        ║"
echo "  ║                                                       ║"
echo "  ║   Quantum-Enhanced Drug Discovery Platform Demo       ║"
echo "  ║                                                       ║"
echo "  ╚═══════════════════════════════════════════════════════╝"
echo ""

# Activate virtual environment
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

cd "$PROJECT_ROOT"

# Parse arguments
MODULES="all"
QUICK=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --modules)
            MODULES="$2"
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

# Run the demo
python scripts/demo/investor_demo.py \
    --modules $MODULES \
    --output-dir "./mvp/demo_results" \
    $QUICK

echo ""
echo "Thank you for watching the QBitaLabs demo!"
echo "Contact: investors@qbitalabs.com"
echo ""
