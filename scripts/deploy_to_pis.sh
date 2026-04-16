#!/bin/bash
# Deploy FL code + data to Raspberry Pis via SSH

set -e

echo "=========================================="
echo "  Deploying FL-Blockchain-EVM to Pis"
echo "=========================================="

# Configuration
PI1_USER=${PI1_USER:-"pi"}
PI1_HOST=${PI1_HOST:-"fl-client-1.local"}
PI1_SUBJECT=${PI1_SUBJECT:-"1"}
PI2_USER=${PI2_USER:-"pi"}
PI2_HOST=${PI2_HOST:-"fl-client-2.local"}
PI2_SUBJECT=${PI2_SUBJECT:-"2"}

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
DATA_DIR="$PROJECT_DIR/data/MHEALTHDATASET"

# Helper function to deploy to a Pi
deploy_to_pi() {
    local user=$1
    local host=$2
    local subject=$3
    local pi_num=$4

    echo ""
    echo "Deploying to Pi $pi_num ($host)..."

    # Deploy code
    ssh "$user@$host" "rm -rf ~/FL-Blockchain-EVM"
    scp -r "$PROJECT_DIR" "$user@$host:~/FL-Blockchain-EVM"
    ssh "$user@$host" "chmod +x ~/FL-Blockchain-EVM/scripts/*.sh"

    # Deploy data for this subject
    echo "  → Copying subject $subject data..."
    ssh "$user@$host" "mkdir -p ~/FL-Blockchain-EVM/data/MHEALTHDATASET/.npy_cache"

    # Copy .log file if it exists
    if [ -f "$DATA_DIR/mHealth_subject${subject}.log" ]; then
        scp "$DATA_DIR/mHealth_subject${subject}.log" "$user@$host:~/FL-Blockchain-EVM/data/MHEALTHDATASET/"
        echo "  → Copied mHealth_subject${subject}.log"
    fi

    # Copy cached .npy files if they exist
    if [ -f "$DATA_DIR/.npy_cache/s${subject}_data.npy" ]; then
        scp "$DATA_DIR/.npy_cache/s${subject}_data.npy" "$user@$host:~/FL-Blockchain-EVM/data/MHEALTHDATASET/.npy_cache/"
        scp "$DATA_DIR/.npy_cache/s${subject}_labels.npy" "$user@$host:~/FL-Blockchain-EVM/data/MHEALTHDATASET/.npy_cache/"
        echo "  → Copied cached data for subject $subject"
    fi
}

# Deploy to both Pis
deploy_to_pi "$PI1_USER" "$PI1_HOST" "$PI1_SUBJECT" "1"
deploy_to_pi "$PI2_USER" "$PI2_HOST" "$PI2_SUBJECT" "2"

echo ""
echo "=========================================="
echo "  ✓ Deployment complete!"
echo "=========================================="
echo ""
echo "Data status:"
echo "  Pi 1: Subject $PI1_SUBJECT"
echo "  Pi 2: Subject $PI2_SUBJECT"
echo ""
echo "Next: Run setup on each Pi:"
echo "  ssh pi@fl-client-1.local"
echo "  cd FL-Blockchain-EVM"
echo "  bash scripts/setup_pi.sh"
echo ""
