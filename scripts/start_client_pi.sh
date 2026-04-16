#!/bin/bash
# Start Flower Client on Raspberry Pi

set -e

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# Configuration - required parameters
SERVER_ADDRESS=${SERVER_ADDRESS:-""}
CLIENT_ID=${CLIENT_ID:-""}

# Validate inputs
if [ -z "$SERVER_ADDRESS" ]; then
    echo "ERROR: SERVER_ADDRESS not set"
    echo ""
    echo "Usage:"
    echo "  SERVER_ADDRESS=192.168.1.100:8080 CLIENT_ID=0 bash scripts/start_client_pi.sh"
    echo ""
    exit 1
fi

if [ -z "$CLIENT_ID" ]; then
    echo "ERROR: CLIENT_ID not set (0 or 1)"
    echo ""
    echo "Usage:"
    echo "  SERVER_ADDRESS=192.168.1.100:8080 CLIENT_ID=0 bash scripts/start_client_pi.sh"
    echo ""
    exit 1
fi

echo "=========================================="
echo "  Starting FL Client (Pi)"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Server: $SERVER_ADDRESS"
echo "  Client ID: $CLIENT_ID"
echo "  Subject (data): $((CLIENT_ID + 1))"
echo ""
echo "=========================================="
echo ""

cd "$PROJECT_ROOT"

# Activate venv
source venv/bin/activate

# Run client
flower-client-app \
    --server-address=$SERVER_ADDRESS \
    --client-id=$CLIENT_ID \
    --insecure
