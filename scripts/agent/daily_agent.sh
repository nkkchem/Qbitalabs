#!/bin/bash
# QBitaLabs Daily Agent Runner
# Wrapper script for automated nightly execution

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$PROJECT_ROOT/mvp/logs"
VENV_DIR="$PROJECT_ROOT/.venv"

# Create log directory
mkdir -p "$LOG_DIR"

# Timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/daily_agent_$TIMESTAMP.log"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "============================================"
log "QBitaLabs Daily Agent Starting"
log "Project Root: $PROJECT_ROOT"
log "============================================"

# Activate virtual environment if exists
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    log "Virtual environment activated"
fi

# Change to project root
cd "$PROJECT_ROOT"

# Parse arguments
SCHEDULE="nightly"
TASK=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            SCHEDULE="quick"
            shift
            ;;
        --hourly)
            SCHEDULE="hourly"
            shift
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Run the daily agent
log "Running daily agent (schedule: $SCHEDULE)"

if [ -n "$TASK" ]; then
    python scripts/agent/daily_agent.py --base-dir "$PROJECT_ROOT" --task "$TASK" 2>&1 | tee -a "$LOG_FILE"
else
    python scripts/agent/daily_agent.py --base-dir "$PROJECT_ROOT" --schedule "$SCHEDULE" 2>&1 | tee -a "$LOG_FILE"
fi

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    log "Daily agent completed successfully"
else
    log "Daily agent completed with errors (exit code: $EXIT_CODE)"
fi

log "Log saved to: $LOG_FILE"
log "============================================"

exit $EXIT_CODE
