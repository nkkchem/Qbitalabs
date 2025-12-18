#!/bin/bash
# QBitaLabs Environment Setup Script
# Usage: ./scripts/setup_env.sh [--dev] [--quantum] [--neuromorphic]

set -e

echo "========================================"
echo "  QBitaLabs Environment Setup"
echo "  Quantum-Bio Swarm Intelligence"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
DEV_MODE=false
QUANTUM=false
NEUROMORPHIC=false

for arg in "$@"; do
    case $arg in
        --dev)
            DEV_MODE=true
            shift
            ;;
        --quantum)
            QUANTUM=true
            shift
            ;;
        --neuromorphic)
            NEUROMORPHIC=true
            shift
            ;;
        --all)
            DEV_MODE=true
            QUANTUM=true
            NEUROMORPHIC=true
            shift
            ;;
        --help)
            echo "Usage: ./setup_env.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dev          Install development dependencies"
            echo "  --quantum      Install quantum computing backends"
            echo "  --neuromorphic Install neuromorphic computing dependencies"
            echo "  --all          Install everything"
            echo "  --help         Show this help message"
            exit 0
            ;;
    esac
done

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}Error: Python $REQUIRED_VERSION or higher is required. Found $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}Python $PYTHON_VERSION detected ✓${NC}"

# Create virtual environment if it doesn't exist
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "\n${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv $VENV_DIR
    echo -e "${GREEN}Virtual environment created ✓${NC}"
else
    echo -e "${GREEN}Virtual environment already exists ✓${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source $VENV_DIR/bin/activate

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install base dependencies
echo -e "\n${YELLOW}Installing base dependencies...${NC}"
pip install -e .

# Install development dependencies
if [ "$DEV_MODE" = true ]; then
    echo -e "\n${YELLOW}Installing development dependencies...${NC}"
    pip install -e ".[dev]"

    # Install pre-commit hooks
    echo -e "\n${YELLOW}Setting up pre-commit hooks...${NC}"
    pre-commit install
fi

# Install quantum dependencies
if [ "$QUANTUM" = true ]; then
    echo -e "\n${YELLOW}Installing quantum computing dependencies...${NC}"
    pip install qiskit qiskit-aer pennylane cirq
fi

# Install neuromorphic dependencies
if [ "$NEUROMORPHIC" = true ]; then
    echo -e "\n${YELLOW}Installing neuromorphic computing dependencies...${NC}"
    pip install brian2 nengo
fi

# Create necessary directories
echo -e "\n${YELLOW}Creating data directories...${NC}"
mkdir -p data/raw data/processed data/models
mkdir -p logs
mkdir -p .cache

# Set up configuration
if [ ! -f "configs/local.yaml" ]; then
    echo -e "\n${YELLOW}Creating local configuration...${NC}"
    cp configs/default.yaml configs/local.yaml
    echo -e "${GREEN}Local config created at configs/local.yaml ✓${NC}"
fi

# Verify installation
echo -e "\n${YELLOW}Verifying installation...${NC}"
python -c "import qbitalabs; print(f'QBitaLabs v{qbitalabs.__version__} installed successfully!')"

echo -e "\n${GREEN}========================================"
echo "  Setup Complete!"
echo "========================================"
echo -e "${NC}"
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest tests/"
echo ""
echo "To start the API server:"
echo "  uvicorn qbitalabs.api:app --reload"
echo ""
echo "For more information, visit: https://docs.qbitalabs.com"
