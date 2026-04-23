#!/usr/bin/env bash
# fl.sh — Federated Learning orchestrator for N Raspberry Pi 4s
#
#   ./fl.sh setup              One-time setup: deploy code + data, install deps
#   ./fl.sh train              Run FL (blockchain variant from .env)
#   ./fl.sh train baseline     Force baseline blockchain (BLOCKCHAIN_OPTIMIZED=0)
#   ./fl.sh train optimized    Force optimized blockchain (BLOCKCHAIN_OPTIMIZED=1)
#   ./fl.sh stop               Stop all FL processes on laptop and Pis
#   ./fl.sh logs               Stream live logs from all devices
#   ./fl.sh status             Show training progress and system status
#
# ── TO ADD MORE PIs: just add their hostname to PI_HOSTS below ──
#
# Configuration via environment variables:
#   PI_USER=pi         SSH user on all Pis (default: pi)
#   PI_PASSWORD=...    SSH password for initial key bootstrap (default: 123123)
#   SSH_KEY_PATH=...   Local private key path (default: ~/.ssh/id_ed25519)
#   NUM_ROUNDS=10      Training rounds (default: 10)
#   LOCAL_EPOCHS=5     Local training epochs per round (default: 5)
#   LR=0.002           Learning rate (default: 0.002)
#   BATCH_SIZE=64      Batch size (default: 64)
#   SERVER_IP=...      Override auto-detected laptop IP

set -euo pipefail

# ── Pi hostnames — ADD NEW PIs HERE ───────────────────────────
PI_HOSTS=(
    "raspberrypi1.local"   
    "raspberrypi.local"    
    "raspberrypi2.local" 
    "raspberrypi3.local" 
    "raspberrypi4.local"
    "raspberrypi5.local"
    "raspberrypi6.local"
    "raspberrypi7.local"
    # "raspberrypi8.local"
    # "raspberrypi9.local"
)

# ── Training defaults ──────────────────────────────────────────
PI_USER="${PI_USER:-pi}"
PI_PASSWORD="${PI_PASSWORD:-123123}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_ed25519}"
NUM_ROUNDS="${NUM_ROUNDS:-10}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
LR="${LR:-0.002}"
BATCH_SIZE="${BATCH_SIZE:-64}"

# Derived automatically — do not change
NUM_PARTITIONS="${#PI_HOSTS[@]}"

# ── Flower ports ───────────────────────────────────────────────
readonly SL_SERVERAPPIO_PORT=9091
readonly SL_FLEET_PORT=9092
readonly SL_CONTROL_PORT=9093
readonly SN_CLIENTAPPIO_PORT=9094
readonly DASHBOARD_PORT=8080

readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_DIR="/tmp/fl_logs"

# ── Colours ────────────────────────────────────────────────────
R='\033[0;31m'; G='\033[0;32m'; Y='\033[1;33m'
C='\033[0;36m'; B='\033[1m'; N='\033[0m'

ts()   { date '+%Y-%m-%d %H:%M:%S'; }
info() { echo -e "${G}[$(ts)] INFO ${N} $*"; }
warn() { echo -e "${Y}[$(ts)] WARN ${N} $*"; }
err()  { echo -e "${R}[$(ts)] ERROR${N} $*" >&2; }
step() { echo -e "\n${C}${B}══ $* ══${N}  [$(ts)]"; }
die()  { err "$*"; exit 1; }

pi_ssh() {
    local host="$1"; shift
    ssh -o StrictHostKeyChecking=no \
        -o ConnectTimeout=10 \
        -o BatchMode=yes \
        -o ServerAliveInterval=30 \
        "${PI_USER}@${host}" "$@"
}

    pi_ssh_bootstrap() {
        local host="$1"; shift
        ssh -o StrictHostKeyChecking=no \
        -o ConnectTimeout=10 \
        -o ServerAliveInterval=30 \
        "${PI_USER}@${host}" "$@"
    }

pi_scp() {
    scp -o StrictHostKeyChecking=no \
        -o ConnectTimeout=10 \
        -o BatchMode=yes \
        "$@"
}

check_pi() { pi_ssh "$1" "echo ok" >/dev/null 2>&1; }

ensure_local_ssh_key() {
    local pub_key="${SSH_KEY_PATH}.pub"

    [ -d "$(dirname "$SSH_KEY_PATH")" ] || mkdir -p "$(dirname "$SSH_KEY_PATH")"
    chmod 700 "$(dirname "$SSH_KEY_PATH")" 2>/dev/null || true

    if [ -f "$SSH_KEY_PATH" ] && [ -f "$pub_key" ]; then
        return 0
    fi

    info "No SSH key found at $SSH_KEY_PATH. Generating one..."
    ssh-keygen -t ed25519 -N "" -f "$SSH_KEY_PATH" >/dev/null
    info "  ✓ Created keypair: $SSH_KEY_PATH"
}

bootstrap_pi_key() {
    local host="$1"
    local pub_key="${SSH_KEY_PATH}.pub"

    check_pi "$host" && { info "  ✓ $host (key auth already works)"; return 0; }

    info "  ↻ Bootstrapping SSH key on $host"

    if command -v ssh-copy-id >/dev/null 2>&1; then
        if [ -n "$PI_PASSWORD" ] && command -v sshpass >/dev/null 2>&1; then
            sshpass -p "$PI_PASSWORD" ssh-copy-id \
                -i "$pub_key" \
                -o StrictHostKeyChecking=no \
                -o ConnectTimeout=10 \
                "${PI_USER}@${host}" >/dev/null 2>&1 \
                || die "Failed to copy SSH key to $host using PI_PASSWORD"
        else
            ssh-copy-id \
                -i "$pub_key" \
                -o StrictHostKeyChecking=no \
                -o ConnectTimeout=10 \
                "${PI_USER}@${host}" \
                || die "Failed to copy SSH key to $host"
        fi
    elif [ -n "$PI_PASSWORD" ] && command -v sshpass >/dev/null 2>&1; then
        cat "$pub_key" | sshpass -p "$PI_PASSWORD" ssh \
            -o StrictHostKeyChecking=no \
            -o ConnectTimeout=10 \
            "${PI_USER}@${host}" \
            "mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && cat >> ~/.ssh/authorized_keys" \
            || die "Failed to append SSH key on $host"
    else
        die "Cannot bootstrap SSH key for $host: install ssh-copy-id, or install sshpass and set PI_PASSWORD"
    fi

    check_pi "$host" \
        && info "  ✓ $host (key bootstrap complete)" \
        || die "Key bootstrap finished but passwordless SSH still fails for $host"
}

ensure_ssh_access_all_pis() {
    ensure_local_ssh_key
    for pi in "${PI_HOSTS[@]}"; do
        bootstrap_pi_key "$pi"
    done
}

server_ip() {
    local ip=""
    for iface in en0 en1 en2 en3; do
        ip=$(ipconfig getifaddr "$iface" 2>/dev/null) && [ -n "$ip" ] && echo "$ip" && return
    done
    ip=$(ip route get 8.8.8.8 2>/dev/null | awk '/src/{print $7; exit}')
    [ -n "$ip" ] && echo "$ip" && return
    hostname -I 2>/dev/null | awk '{print $1}'
}

activate_venv() {
    cd "$PROJECT_DIR"
    [ -d "venv" ] && source venv/bin/activate || true
}

# ─────────────────────────────────────────────────────────────
#  SETUP — deploy code + data, install deps on all Pis
# ─────────────────────────────────────────────────────────────
cmd_setup() {
    step "FL SETUP — ${NUM_PARTITIONS} Pi(s)"

    step "Ensuring SSH key-based access"
    ensure_ssh_access_all_pis

    info "Checking SSH connectivity..."
    for pi in "${PI_HOSTS[@]}"; do
        check_pi "$pi" \
            && info "  ✓ $pi" \
            || die "Cannot reach $pi.\n  Enable SSH: sudo raspi-config → Interface Options → SSH\n  Verify: ssh ${PI_USER}@${pi}"
    done

    step "Deploying to ${NUM_PARTITIONS} Pi(s) in parallel"
    mkdir -p "$LOG_DIR"

    _deploy() {
        local pi_host="$1"
        local logf="$LOG_DIR/setup_${pi_host}.log"
        _plog() { echo "[$(ts)] [${pi_host}] $*"; }
        {
            _plog "── Setup start ──"

            _plog "Syncing source code..."
            pi_ssh "$pi_host" "rm -rf ~/FL-Blockchain-EVM-tmp" 2>/dev/null || true
            tar -czf - \
                --exclude='venv' --exclude='.git' --exclude='__pycache__' \
                --exclude='*.pyc' --exclude='outputs' --exclude='.env' \
                --exclude='final_model.pt' --exclude='*.bak' \
                -C "$(dirname "$PROJECT_DIR")" "$(basename "$PROJECT_DIR")" \
            | pi_ssh "$pi_host" \
                "cd ~ \
                 && tar -xzf - --warning=no-unknown-keyword 2>/dev/null \
                 && dirname_tar=\$(ls -d ~/FL-Blockchain-EVM* 2>/dev/null | head -1) \
                 && [ -d ~/FL-Blockchain-EVM ] || mv \"\$dirname_tar\" ~/FL-Blockchain-EVM \
                 && chmod +x ~/FL-Blockchain-EVM/fl.sh \
                 && echo \"[\$(date '+%Y-%m-%d %H:%M:%S')] Code synced\""

            _plog "Syncing MHEALTH data..."
            local data_src="$PROJECT_DIR/data/MHEALTHDATASET"
            pi_ssh "$pi_host" "mkdir -p ~/FL-Blockchain-EVM/data/MHEALTHDATASET/.npy_cache"
            # Copy real log files if present
            if ls "$data_src"/mHealth_subject*.log 1>/dev/null 2>&1; then
                local subject_count
                subject_count=$(ls "$data_src"/mHealth_subject*.log 2>/dev/null | wc -l | tr -d ' ')
                _plog "Copying ${subject_count} subject files..."
                pi_scp -q "$data_src"/mHealth_subject*.log \
                    "${PI_USER}@${pi_host}:~/FL-Blockchain-EVM/data/MHEALTHDATASET/"
                _plog "Copied ${subject_count} subject files."
            fi
            # Copy npy cache if present
            if ls "$data_src/.npy_cache"/s*.npy 1>/dev/null 2>&1; then
                local npy_count
                npy_count=$(ls "$data_src/.npy_cache"/s*.npy 2>/dev/null | wc -l | tr -d ' ')
                _plog "Copying ${npy_count} npy cache files..."
                pi_scp -q "$data_src/.npy_cache"/s*.npy \
                    "${PI_USER}@${pi_host}:~/FL-Blockchain-EVM/data/MHEALTHDATASET/.npy_cache/"
                _plog "Copied ${npy_count} npy cache files."
            fi

            [ -f "$PROJECT_DIR/.env" ] && \
                pi_scp -q "$PROJECT_DIR/.env" "${PI_USER}@${pi_host}:~/FL-Blockchain-EVM/.env" && \
                _plog ".env copied."

            _plog "Installing Python dependencies (may take 15-30 min first time)..."
            pi_ssh "$pi_host" bash << 'PISETUP'
set -e
cd ~/FL-Blockchain-EVM
_l() { echo "[$(date '+%H:%M:%S')] $*"; }

_l "apt-get update..."
sudo apt-get update -qq 2>/dev/null || true
sudo apt-get install -y -qq build-essential python3 python3-venv python3-dev \
    libopenblas-dev libblas-dev liblapack-dev git curl 2>/dev/null || true

_l "Creating venv..."
[ -d "venv" ] || python3 -m venv venv
source venv/bin/activate

_l "Upgrading pip..."
pip install --upgrade pip setuptools wheel --quiet

_l "Checking existing FL dependencies..."
if python3 - << 'VCHK' >/dev/null 2>&1
import importlib
for name in ("flwr", "torch", "numpy", "sklearn"):
    importlib.import_module(name)
VCHK
then
    _l "Dependencies already present, skipping pip install"
else
    _l "Installing fl dependencies (streaming pip output)..."
    pip install -r requirements-pi.txt 2>&1 | while IFS= read -r line; do
        _l "[pip] $line"
    done
    rc=${PIPESTATUS[0]}
    [ "$rc" -eq 0 ] || { _l "pip install failed with code $rc"; exit "$rc"; }
fi

_l "Verifying..."
python3 - << 'V'
import sys
for pkg in [("flwr","__version__"),("torch","__version__"),
            ("numpy","__version__"),("sklearn","__version__")]:
    try:
        m = __import__(pkg[0])
        print(f"  ✓ {pkg[0]:<12} {getattr(m, pkg[1], '?')}")
    except ImportError:
        print(f"  ✗ {pkg[0]:<12} MISSING")
        sys.exit(1)
V

which flower-supernode >/dev/null 2>&1 \
    && _l "  ✓ flower-supernode: $(which flower-supernode)" \
    || { _l "  ✗ flower-supernode not found"; exit 1; }

_l "Setup COMPLETE on $(hostname)"
PISETUP
            _plog "✓ setup done"
        } 2>&1 | tee "$logf"
    }

    # Launch all deployments in parallel
    local pids=()
    for pi in "${PI_HOSTS[@]}"; do
        _deploy "$pi" &
        pids+=($!)
    done

    local ok=true
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { err "${PI_HOSTS[$i]} failed → $LOG_DIR/setup_${PI_HOSTS[$i]}.log"; ok=false; }
    done
    [ "$ok" = "true" ] || exit 1

    step "SETUP COMPLETE"
    for i in "${!PI_HOSTS[@]}"; do
        info "  Pi $i (${PI_HOSTS[$i]}) — partition $i"
    done
    info "  Run './fl.sh train' to start training."
}

# ─────────────────────────────────────────────────────────────
#  TRAIN — full FL run: SuperLink + dashboard + SuperNodes + flwr run
# ─────────────────────────────────────────────────────────────
cmd_train() {
    # ── Blockchain variant ────────────────────────────────────
    local _variant="${1:-}"
    case "$_variant" in
        baseline)
            export BLOCKCHAIN_OPTIMIZED=0
            export EXPERIMENT_VARIANT=baseline
            ;;
        optimized)
            export BLOCKCHAIN_OPTIMIZED=1
            export EXPERIMENT_VARIANT=optimized
            ;;
        "")
            # Use whatever BLOCKCHAIN_OPTIMIZED / EXPERIMENT_VARIANT are in .env
            BLOCKCHAIN_OPTIMIZED="${BLOCKCHAIN_OPTIMIZED:-0}"
            EXPERIMENT_VARIANT="${EXPERIMENT_VARIANT:-experiment}"
            export BLOCKCHAIN_OPTIMIZED EXPERIMENT_VARIANT
            ;;
        *)
            die "Unknown variant '$_variant'. Use: baseline  optimized"
            ;;
    esac

    local _bc_label
    [ "$BLOCKCHAIN_OPTIMIZED" = "1" ] && _bc_label="OPTIMIZED" || _bc_label="BASELINE"

    step "FL TRAINING — $NUM_ROUNDS rounds, $NUM_PARTITIONS partitions  [$_bc_label]"

    local SRV_IP="${SERVER_IP:-$(server_ip)}"
    [ -n "$SRV_IP" ] || die "Could not detect laptop IP.\nSet: SERVER_IP=192.168.x.x ./fl.sh train"

    info "  Server IP       : $SRV_IP"
    for i in "${!PI_HOSTS[@]}"; do
        info "  Pi $i (part. $i)  : ${PI_HOSTS[$i]}"
    done
    info "  Rounds          : $NUM_ROUNDS"
    info "  LR / Epochs     : $LR / $LOCAL_EPOCHS"
    info "  Blockchain      : $_bc_label  (EXPERIMENT_VARIANT=$EXPERIMENT_VARIANT)"
    info "  Flower ports    : Fleet=$SL_FLEET_PORT  Control=$SL_CONTROL_PORT"
    info "  Dashboard       : http://localhost:$DASHBOARD_PORT/monitor"

    for pi in "${PI_HOSTS[@]}"; do
        check_pi "$pi" || die "$pi unreachable — run './fl.sh setup' first"
        info "  ✓ $pi reachable"
    done

    activate_venv
    mkdir -p "$LOG_DIR" "outputs"
    rm -f "$LOG_DIR"/*.log outputs/results.json outputs/fl_server.log
    for f in training superlink dashboard; do : > "$LOG_DIR/${f}.log"; done
    : > "outputs/fl_server.log"

    # ── Cleanup on Ctrl+C or exit ────────────────────────────
    _SL_PID="" _DASH_PID=""
    cleanup() {
        step "SHUTDOWN"
        [ -n "$_SL_PID"   ] && kill "$_SL_PID"   2>/dev/null && info "  ✓ SuperLink stopped"   || true
        [ -n "$_DASH_PID" ] && kill "$_DASH_PID"  2>/dev/null && info "  ✓ Dashboard stopped"   || true
        pkill -f "flower-superlink" 2>/dev/null || true
        for pi in "${PI_HOSTS[@]}"; do
            pi_ssh "$pi" "pkill -f flower-supernode 2>/dev/null; true" 2>/dev/null &
        done
        wait; info "  ✓ Pi processes stopped"
    }
    trap cleanup EXIT

    # ── 1. Dashboard ─────────────────────────────────────────
    step "1/4  Dashboard"
    python run_dashboard.py > "$LOG_DIR/dashboard.log" 2>&1 &
    _DASH_PID=$!
    sleep 2
    kill -0 "$_DASH_PID" 2>/dev/null \
        && info "  ✓ Dashboard running → http://localhost:$DASHBOARD_PORT/monitor" \
        || { warn "  Dashboard failed (training continues)"; warn "  Log: $LOG_DIR/dashboard.log"; }

    # ── 2. SuperLink ─────────────────────────────────────────
    step "2/4  Flower SuperLink"
    pkill -f "flower-superlink" 2>/dev/null || true; sleep 1

    export FL_DATA_DIR="$PROJECT_DIR/data/MHEALTHDATASET"
    cd "$PROJECT_DIR"
    flower-superlink \
        --insecure \
        --serverappio-api-address "0.0.0.0:${SL_SERVERAPPIO_PORT}" \
        --fleet-api-address       "0.0.0.0:${SL_FLEET_PORT}" \
        --control-api-address     "0.0.0.0:${SL_CONTROL_PORT}" \
        > "$LOG_DIR/superlink.log" 2>&1 &
    _SL_PID=$!
    sleep 3

    kill -0 "$_SL_PID" 2>/dev/null \
        || { err "SuperLink failed to start"; cat "$LOG_DIR/superlink.log"; exit 1; }
    info "  ✓ SuperLink PID=$_SL_PID"
    info "    Fleet API    (supernodes) : $SRV_IP:$SL_FLEET_PORT"
    info "    Control API  (flwr run)   : $SRV_IP:$SL_CONTROL_PORT"

    # ── 3. SuperNodes on all Pis ─────────────────────────────
    step "3/4  SuperNodes on ${NUM_PARTITIONS} Pi(s)"

    _start_supernode() {
        local pi_host="$1"
        local part_id="$2"
        local logf="$LOG_DIR/supernode_${part_id}.log"

        local script
        script=$(cat << PIRUN
set -e
LOGF="/tmp/fl_client_${part_id}.log"
: > "\$LOGF"
_l() { echo "[\$(date '+%Y-%m-%d %H:%M:%S')] \$*" | tee -a "\$LOGF"; }

_l "════════════════════════════════════════════════════"
_l "  FL SuperNode — Partition ${part_id} / ${NUM_PARTITIONS}"
_l "  Host      : \$(hostname)  (${pi_host})"
_l "  Server    : ${SRV_IP}:${SL_FLEET_PORT}"
_l "  FL_DATA_DIR: \$HOME/FL-Blockchain-EVM/data/MHEALTHDATASET"
_l "════════════════════════════════════════════════════"

cd ~/FL-Blockchain-EVM
source venv/bin/activate

pkill -f flower-supernode 2>/dev/null && _l "Killed stale supernode" || true
sleep 1

export FL_DATA_DIR="\$HOME/FL-Blockchain-EVM/data/MHEALTHDATASET"

_l "Starting flower-supernode..."
flower-supernode \\
    --insecure \\
    --superlink "${SRV_IP}:${SL_FLEET_PORT}" \\
    --node-config "partition-id=${part_id} num-partitions=${NUM_PARTITIONS}" \\
    --clientappio-api-address "0.0.0.0:${SN_CLIENTAPPIO_PORT}" \\
    2>&1 | while IFS= read -r line; do
        echo "[\$(date '+%Y-%m-%d %H:%M:%S')] \$line" | tee -a "\$LOGF"
    done
_l "SuperNode exited."
PIRUN
)
        echo "$script" | pi_ssh "$pi_host" bash > "$logf" 2>&1 &
        sleep 2
        info "  ✓ SuperNode ${part_id} started on ${pi_host}"
        info "    Remote log : ssh ${PI_USER}@${pi_host} tail -f /tmp/fl_client_${part_id}.log"
        info "    Local copy : $logf"
    }

    for i in "${!PI_HOSTS[@]}"; do
        _start_supernode "${PI_HOSTS[$i]}" "$i"
    done

    info ""
    info "  Waiting 15s for SuperNodes to register with SuperLink..."
    sleep 15

    local connected
    connected=$(grep -Eic "New node|registered|activate|pullmessages|supernode" "$LOG_DIR/superlink.log" 2>/dev/null || true)
    connected="${connected:-0}"
    [ "$connected" -ge 1 ] \
        && info "  ✓ SuperLink reports activity (connections detected)" \
        || warn "  No connection events yet in SuperLink log — SuperNodes may still be connecting"

    # ── 4. Run training ──────────────────────────────────────
    step "4/4  FL Training  (flwr run)"

    if ! grep -q '^\[tool\.flwr\.federations\]' pyproject.toml; then
        cat >> pyproject.toml << 'EOF'

[tool.flwr.federations]
default = "remote-federation"
EOF
        info "  pyproject.toml → added [tool.flwr.federations]"
    fi

    if ! grep -q '^\[tool\.flwr\.federations\.remote-federation\]' pyproject.toml; then
        cat >> pyproject.toml << EOF

[tool.flwr.federations.remote-federation]
address = "${SRV_IP}:${SL_CONTROL_PORT}"
insecure = true
EOF
        info "  pyproject.toml → added [tool.flwr.federations.remote-federation]"
    fi

    # Update ~/.flwr/config.toml remote-federation address (Flower 1.29+)
    local flwr_cfg="$HOME/.flwr/config.toml"
    if [ -f "$flwr_cfg" ]; then
        python3 - "$flwr_cfg" "${SRV_IP}:${SL_CONTROL_PORT}" << 'PY'
import sys, re
cfg_path, new_addr = sys.argv[1], sys.argv[2]
with open(cfg_path) as f:
    content = f.read()
content = re.sub(
    r'(\[superlink\.remote-federation\][^\[]*address\s*=\s*")[^"]*(")',
    lambda m: m.group(1) + new_addr + m.group(2),
    content, flags=re.DOTALL
)
with open(cfg_path, 'w') as f:
    f.write(content)
PY
        info "  ~/.flwr/config.toml → remote-federation.address = ${SRV_IP}:${SL_CONTROL_PORT}"
    fi

    sed -i.bak \
        -e "/^\[tool\.flwr\.federations\.remote-federation\]/,/^\[/ s|^address = \".*\"|address = \"${SRV_IP}:${SL_CONTROL_PORT}\"|" \
        -e "/^\[tool\.flwr\.federations\.remote-federation\]/,/^\[/ s|^insecure = .*|insecure = true|" \
        pyproject.toml
    info "  pyproject.toml → remote-federation.address = ${SRV_IP}:${SL_CONTROL_PORT}"

    # Update [tool.flwr.app.config]
    sed -i.bak \
        -e "s/^num-server-rounds = .*/num-server-rounds = ${NUM_ROUNDS}/" \
        -e "s/^lr = .*/lr = ${LR}/" \
        -e "s/^local-epochs = .*/local-epochs = ${LOCAL_EPOCHS}/" \
        -e "s/^batch-size = .*/batch-size = ${BATCH_SIZE}/" \
        pyproject.toml
    info "  pyproject.toml → rounds=$NUM_ROUNDS  lr=$LR  local-epochs=$LOCAL_EPOCHS  batch-size=$BATCH_SIZE"
    info ""
    info "  Streaming training logs below (Ctrl+C to stop):"
    info "  ServerApp log : outputs/fl_server.log"
    info "  SuperLink log : $LOG_DIR/superlink.log"
    info ""

    export NUM_ROUNDS LOCAL_EPOCHS LR BATCH_SIZE NUM_PARTITIONS
    export FL_DATA_DIR="$PROJECT_DIR/data/MHEALTHDATASET"

    flwr run . remote-federation \
        --run-config "num-server-rounds=${NUM_ROUNDS} lr=${LR} local-epochs=${LOCAL_EPOCHS} batch-size=${BATCH_SIZE} fraction-train=1.0" \
        --stream \
        2>&1 | tee "$LOG_DIR/training.log"

    local rc="${PIPESTATUS[0]}"
    if [ "$rc" -eq 0 ]; then
        step "TRAINING COMPLETE"
        info "  Results    : outputs/results.json"
        info "  Final model: final_model.pt"
        info "  Dashboard  : http://localhost:$DASHBOARD_PORT/monitor"
        info "  Server log : outputs/fl_server.log"
        activate_venv 2>/dev/null || true
        python3 - << 'PY' 2>/dev/null || true
import json, os
rounds = []
path = "outputs/results.json"
if os.path.exists(path):
    with open(path) as f:
        for line in f:
            try:
                o = json.loads(line.strip())
                if isinstance(o,dict) and o.get("type")=="global":
                    rounds.append(o)
            except Exception: pass
if rounds:
    r = rounds[-1]
    print(f"  Final (round {r['round']}): "
          f"acc={r.get('accuracy',0):.4f}  "
          f"f1={r.get('f1_macro',0):.4f}  "
          f"auc={r.get('auc_macro',0):.4f}  "
          f"loss={r.get('loss',0):.4f}")
else:
    print("  (no global metrics yet — check outputs/fl_server.log)")
PY
    else
        err "flwr run exited with code $rc"
        err "Diagnose: tail $LOG_DIR/training.log"
        err "Server  : tail outputs/fl_server.log"
    fi
    return "$rc"
}

# ─────────────────────────────────────────────────────────────
#  STOP
# ─────────────────────────────────────────────────────────────
cmd_stop() {
    step "Stopping all FL processes"
    pkill -f "flower-superlink"  2>/dev/null && info "  ✓ SuperLink"   || info "  - SuperLink not running"
    pkill -f "flwr run"          2>/dev/null && info "  ✓ flwr run"    || true
    pkill -f "run_dashboard"     2>/dev/null && info "  ✓ Dashboard"   || info "  - Dashboard not running"
    pkill -f "uvicorn"           2>/dev/null && info "  ✓ uvicorn"     || true
    for pi in "${PI_HOSTS[@]}"; do
        if check_pi "$pi" 2>/dev/null; then
            pi_ssh "$pi" "pkill -f flower-supernode 2>/dev/null; true" 2>/dev/null \
                && info "  ✓ $pi SuperNode stopped" || true
        else
            warn "  $pi unreachable — skipped"
        fi
    done
    info ""; info "✓ Done."
}

# ─────────────────────────────────────────────────────────────
#  LOGS — live log aggregation from all devices
# ─────────────────────────────────────────────────────────────
cmd_logs() {
    step "Live Logs — all devices  (Ctrl+C to exit)"
    mkdir -p "$LOG_DIR"
    for f in training superlink dashboard; do touch "$LOG_DIR/${f}.log"; done
    touch "outputs/fl_server.log" 2>/dev/null || true

    for i in "${!PI_HOSTS[@]}"; do
        local pi_host="${PI_HOSTS[$i]}"
        if check_pi "$pi_host" 2>/dev/null; then
            (pi_ssh "$pi_host" \
                "tail -n 40 -f /tmp/fl_client_${i}.log 2>/dev/null || echo 'no log yet'" \
                2>/dev/null \
                | sed "s/^/${C}[Pi${i} ${pi_host}]${N} /") &
        else
            warn "  $pi_host unreachable"
        fi
    done

    tail -n 20 -f \
        "$LOG_DIR/training.log" \
        "$LOG_DIR/superlink.log" \
        "outputs/fl_server.log" \
    2>/dev/null \
    | awk -v g="$G" -v y="$Y" -v c="$C" -v n="$N" '
        /==> .* training\.log/   { src=g"[TRAINING]  "n; next }
        /==> .* superlink\.log/  { src=y"[SUPERLINK] "n; next }
        /==> .* fl_server\.log/  { src=c"[SERVER]    "n; next }
        { print src $0 }
    ' &

    wait
}

# ─────────────────────────────────────────────────────────────
#  STATUS
# ─────────────────────────────────────────────────────────────
cmd_status() {
    step "FL System Status  [$(ts)]"
    echo ""
    echo "─── Laptop ──────────────────────────────────────────"
    pgrep -f "flower-superlink" >/dev/null 2>&1 \
        && echo "  ✓ SuperLink    running" || echo "  ✗ SuperLink    stopped"
    pgrep -f "flwr run"         >/dev/null 2>&1 \
        && echo "  ✓ flwr run     running" || echo "  ✗ flwr run     stopped"
    pgrep -f "run_dashboard"    >/dev/null 2>&1 \
        && echo "  ✓ Dashboard    running → http://localhost:${DASHBOARD_PORT}/monitor" \
        || echo "  ✗ Dashboard    stopped"
    echo ""
    echo "─── Pis (${NUM_PARTITIONS} total) ───────────────────────────────────"
    for i in "${!PI_HOSTS[@]}"; do
        local pi_host="${PI_HOSTS[$i]}"
        if check_pi "$pi_host" 2>/dev/null; then
            local st
            st=$(pi_ssh "$pi_host" \
                "pgrep -fa flower-supernode 2>/dev/null | head -1 || echo 'stopped'" 2>/dev/null)
            echo "$st" | grep -q "flower-supernode" \
                && echo "  ✓ Pi${i} ($pi_host)  SuperNode running" \
                || echo "  ✗ Pi${i} ($pi_host)  SuperNode stopped"
        else
            echo "  ✗ Pi${i} ($pi_host)  UNREACHABLE"
        fi
    done
    echo ""
    echo "─── Training ────────────────────────────────────────"
    activate_venv 2>/dev/null || true
    python3 - << 'PY' 2>/dev/null || echo "  (no results yet)"
import json, os
path = "outputs/results.json"
if not os.path.exists(path):
    print("  No results — run './fl.sh train'")
else:
    rounds = []
    with open(path) as f:
        for line in f:
            try:
                o = json.loads(line.strip())
                if isinstance(o,dict) and o.get("type")=="global":
                    rounds.append(o)
            except Exception: pass
    if not rounds:
        print("  Training in progress (no completed rounds yet)...")
    else:
        print(f"  Completed rounds : {len(rounds)}")
        r = rounds[-1]
        print(f"  Latest (round {r['round']:2d}) : "
              f"acc={r.get('accuracy',0):.4f}  "
              f"f1={r.get('f1_macro',0):.4f}  "
              f"auc={r.get('auc_macro',0):.4f}  "
              f"loss={r.get('loss',0):.4f}")
        if len(rounds) > 1:
            da = rounds[-1].get('accuracy',0) - rounds[0].get('accuracy',0)
            df = rounds[-1].get('f1_macro',0)  - rounds[0].get('f1_macro',0)
            print(f"  Trend            : acc{'+' if da>=0 else ''}{da:.4f}  "
                  f"f1{'+' if df>=0 else ''}{df:.4f}")
PY
    echo ""
    echo "─── Logs ────────────────────────────────────────────"
    echo "  ./fl.sh logs"
    for i in "${!PI_HOSTS[@]}"; do
        echo "  ssh ${PI_USER}@${PI_HOSTS[$i]} tail -f /tmp/fl_client_${i}.log"
    done
    echo "  tail -f $LOG_DIR/superlink.log"
    echo "  tail -f outputs/fl_server.log"
    echo ""
}

# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
CMD="${1:-help}"; shift 2>/dev/null || true
case "$CMD" in
    setup)  cmd_setup  ;;
    train)  cmd_train "$@"  ;;
    stop)   cmd_stop   ;;
    logs)   cmd_logs   ;;
    status) cmd_status ;;
    *)
        echo -e ""
        echo -e "${B}FL — Federated Learning (${NUM_PARTITIONS} × Raspberry Pi 4)${N}"
        echo -e ""
        echo -e "Usage: ${B}./fl.sh <command>${N}"
        echo -e ""
        printf "  ${G}%-8s${N}  %s\n" "setup"  "Deploy code + data to all Pis, install all deps"
        printf "  ${G}%-8s${N}  %s\n" "train"  "Start FL (blockchain variant from .env)"
        printf "  ${G}%-10s${N}  %s\n" "train baseline"  "Force baseline blockchain  (BLOCKCHAIN_OPTIMIZED=0)"
        printf "  ${G}%-10s${N}  %s\n" "train optimized" "Force optimized blockchain (BLOCKCHAIN_OPTIMIZED=1)"
        printf "  ${G}%-8s${N}  %s\n" "stop"   "Stop all FL processes on laptop and all Pis"
        printf "  ${G}%-8s${N}  %s\n" "logs"   "Stream live logs from all devices simultaneously"
        printf "  ${G}%-8s${N}  %s\n" "status" "Show training progress and system health"
        echo -e ""
        echo -e "Env overrides:  PI_USER=pi  PI_PASSWORD=123123  SSH_KEY_PATH=~/.ssh/id_ed25519  NUM_ROUNDS=10  LOCAL_EPOCHS=5  BATCH_SIZE=64  LR=0.002  SERVER_IP=..."
        echo -e ""
        echo -e "Pis configured (${NUM_PARTITIONS} total):"
        for i in "${!PI_HOSTS[@]}"; do
            echo -e "  Pi${i} → ${PI_HOSTS[$i]}  (partition $i)"
        done
        echo -e ""
        echo -e "To add a Pi: edit PI_HOSTS array at the top of this file"
        echo -e ""
        echo -e "Workflow: ${C}./fl.sh setup${N} → ${C}./fl.sh train${N}  (open another tab: ${C}./fl.sh logs${N})"
        echo ""
        ;;
esac
