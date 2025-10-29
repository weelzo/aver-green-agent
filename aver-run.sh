#!/bin/bash
# AVER Runner - Runs AVER benchmark with proper environment
# Usage (just like AgentBeats tutorial):
#   ./aver-run.sh scenarios/aver/scenario.toml

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment
source "$SCRIPT_DIR/aver-venv/bin/activate"

# Run AVER CLI
python3 "$SCRIPT_DIR/aver-run" "$@"
