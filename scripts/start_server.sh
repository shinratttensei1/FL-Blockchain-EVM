#!/bin/bash
# Start Flower Server for distributed FL on your laptop

set -e

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# Configuration
NUM_ROUNDS=${NUM_ROUNDS:-5}
SERVER_ADDRESS=${SERVER_ADDRESS:-"0.0.0.0:8080"}

echo "=========================================="
echo "  Starting FL Server"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Rounds: $NUM_ROUNDS"
echo "  Address: $SERVER_ADDRESS"
echo ""
echo "Make sure Pis have the server IP address!"
echo "Get your IP: ifconfig | grep inet"
echo ""
echo "=========================================="
echo ""

cd "$PROJECT_ROOT"

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run server
export NUM_ROUNDS=$NUM_ROUNDS
export SERVER_ADDRESS=$SERVER_ADDRESS
python run_server.py
