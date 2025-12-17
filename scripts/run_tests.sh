#!/bin/bash
# QBitaLabs Test Runner Script
# Usage: ./scripts/run_tests.sh [unit|integration|e2e|all] [--coverage] [--verbose]

set -e

echo "========================================"
echo "  QBitaLabs Test Runner"
echo "========================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default values
TEST_TYPE="all"
COVERAGE=false
VERBOSE=false
PARALLEL=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        unit|integration|e2e|all)
            TEST_TYPE=$arg
            ;;
        --coverage|-c)
            COVERAGE=true
            ;;
        --verbose|-v)
            VERBOSE=true
            ;;
        --parallel|-p)
            PARALLEL=true
            ;;
        --help|-h)
            echo "Usage: ./run_tests.sh [TEST_TYPE] [OPTIONS]"
            echo ""
            echo "Test Types:"
            echo "  unit         Run unit tests only"
            echo "  integration  Run integration tests only"
            echo "  e2e          Run end-to-end tests only"
            echo "  all          Run all tests (default)"
            echo ""
            echo "Options:"
            echo "  --coverage, -c   Generate coverage report"
            echo "  --verbose, -v    Verbose output"
            echo "  --parallel, -p   Run tests in parallel"
            echo "  --help, -h       Show this help message"
            exit 0
            ;;
    esac
done

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Build pytest command
PYTEST_CMD="pytest"

# Add test path based on type
case $TEST_TYPE in
    unit)
        PYTEST_CMD="$PYTEST_CMD tests/unit/"
        echo -e "${YELLOW}Running unit tests...${NC}"
        ;;
    integration)
        PYTEST_CMD="$PYTEST_CMD tests/integration/"
        echo -e "${YELLOW}Running integration tests...${NC}"
        ;;
    e2e)
        PYTEST_CMD="$PYTEST_CMD tests/e2e/"
        echo -e "${YELLOW}Running end-to-end tests...${NC}"
        ;;
    all)
        PYTEST_CMD="$PYTEST_CMD tests/"
        echo -e "${YELLOW}Running all tests...${NC}"
        ;;
esac

# Add options
if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

if [ "$PARALLEL" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -n auto"
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=qbitalabs --cov-report=html --cov-report=term-missing"
fi

# Add standard options
PYTEST_CMD="$PYTEST_CMD --tb=short --color=yes"

echo -e "\n${YELLOW}Executing: $PYTEST_CMD${NC}\n"

# Run tests
$PYTEST_CMD

# Check result
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}========================================"
    echo "  All tests passed! ✓"
    echo "========================================"
    echo -e "${NC}"

    if [ "$COVERAGE" = true ]; then
        echo "Coverage report generated: htmlcov/index.html"
    fi
else
    echo -e "\n${RED}========================================"
    echo "  Some tests failed ✗"
    echo "========================================"
    echo -e "${NC}"
    exit 1
fi
