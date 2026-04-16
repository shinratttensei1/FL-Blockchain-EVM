#!/bin/bash
# Setup script for Raspberry Pi - Installs FL client dependencies
# Run this on each Pi after OS installation

set -e  # Exit on error

echo "=========================================="
echo "  FL-Blockchain-EVM: Raspberry Pi Setup"
echo "=========================================="

# Check if running on Pi
if ! grep -q "Raspberry Pi" /proc/device-tree/model 2>/dev/null; then
    echo "⚠️  Warning: This doesn't appear to be a Raspberry Pi"
    echo "   (But continuing anyway...)"
fi

# Get current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "$SCRIPT_DIR" )"

echo ""
echo "[1/6] Updating system packages..."
sudo apt update
sudo apt upgrade -y

echo ""
echo "[2/6] Installing system dependencies..."
sudo apt install -y \
    build-essential \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    libopenblas-dev \
    libblas-dev \
    liblapack-dev \
    git \
    wget \
    curl

echo ""
echo "[3/6] Creating Python virtual environment..."
cd "$PROJECT_ROOT"
python3.11 -m venv venv
source venv/bin/activate

echo ""
echo "[4/6] Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

echo ""
echo "[5/6] Installing Pi-optimized dependencies..."
pip install -r requirements-pi.txt

echo ""
echo "[6/6] Verifying installation..."
python -c "import torch; import flwr; print('✓ PyTorch and Flower installed')"

echo ""
echo "=========================================="
echo "  ✓ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Get your laptop's IP address:"
echo "     (on laptop: ifconfig | grep inet)"
echo ""
echo "  2. Set CLIENT_ID (0 or 1):"
echo "     export CLIENT_ID=0  # or 1"
echo ""
echo "  3. Run the client:"
echo "     source venv/bin/activate"
echo "     flower-client-app \\"
echo "       --server-address=<LAPTOP_IP>:8080 \\"
echo "       --client-id=\$CLIENT_ID"
echo ""
