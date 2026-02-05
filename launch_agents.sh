#!/bin/bash

# Define the list of agent scripts to launch
AGENTS=(
    "agents/solar.py"
)

# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if ports 8001 to 8005 are in use and close them if necessary
for PORT in {8001..8005}; do
    if lsof -i:$PORT > /dev/null; then
        echo "Port $PORT is in use. Closing the process..."
        fuser -k $PORT/tcp
    fi
done

# Launch each agent in the background
for AGENT in "${AGENTS[@]}"; do
    AGENT_PATH="$SCRIPT_DIR/src/$AGENT"
    if [[ -f "$AGENT_PATH" ]]; then
        echo "Launching $AGENT..."
        python "$AGENT_PATH" &
    else
        echo "Agent script not found: $AGENT"
    fi
done

# Wait for agents to start
sleep 3

# Launch orchestrator at the end
ORCHESTRATOR_PATH="$SCRIPT_DIR/src/agents/orchestrator.py"
if [[ -f "$ORCHESTRATOR_PATH" ]]; then
    echo "Launching agents/orchestrator.py..."
    python "$ORCHESTRATOR_PATH" &
else
    echo "Orchestrator script not found"
fi

# Wait for all background processes to complete
wait