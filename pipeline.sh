#!/usr/bin/env bash
# pipeline.sh — Alternating baseline/optimized FL pipeline
#
#   ./pipeline.sh
#
# Runs 10 baseline and 10 optimized training sessions in alternating order:
#   baseline → optimized → baseline → optimized → ... (×10 each)
#
# Each session is 10 rounds, so:
#   100 total baseline rounds  (10 sessions × 10 rounds)
#   100 total optimized rounds (10 sessions × 10 rounds)
#
# Results are saved automatically to per-run timestamped subdirectories:
#   outputs/baseline_YYYYMMDD_HHMMSS/
#   outputs/optimized_YYYYMMDD_HHMMSS/
#
# Override training params (passed through to fl.sh):
#   NUM_ROUNDS=10 LR=0.002 LOCAL_EPOCHS=1 BATCH_SIZE=256 ./pipeline.sh
#
# Resume support:
#   START_STEP=2 ./pipeline.sh
#     -> starts from the 2nd training in the sequence
#        (optimized run 1, then baseline run 2, ...)
#
# Re-run specific steps only:
#   RUN_STEPS="1 4" ./pipeline.sh
#     -> runs only step 1 (baseline) and step 4 (optimized), in order
#        use this to redo individual failed/incomplete runs
#
# Pinata account routing:
#   Steps 1..TOTAL_RUNS  → PINATA_ACCOUNT=1  (first half, account_1)
#   Steps TOTAL_RUNS+1.. → PINATA_ACCOUNT=2  (second half, account_2)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOTAL_RUNS="${TOTAL_RUNS:-10}"
START_STEP="${START_STEP:-1}"
RUN_STEPS="${RUN_STEPS:-}"      # if set, run only these step indices (e.g. "1 4")
HALF=$((TOTAL_RUNS))            # steps 1..HALF → Pinata account_1, rest → account_2

# ── Colours (same palette as fl.sh) ───────────────────────────
G='\033[0;32m'; Y='\033[1;33m'; C='\033[0;36m'; B='\033[1m'; N='\033[0m'; R='\033[0;31m'
ts()   { date '+%Y-%m-%d %H:%M:%S'; }
info() { echo -e "${G}[$(ts)] INFO${N} $*"; }
warn() { echo -e "${Y}[$(ts)] WARN${N} $*"; }
step() { echo -e "\n${C}${B}══ $* ══${N}  [$(ts)]"; }

# ── On interrupt: stop all FL processes cleanly ────────────────
trap '_rc=$?; echo -e "\n${R}[$(ts)] Pipeline interrupted — stopping FL processes${N}"; "$SCRIPT_DIR/fl.sh" stop 2>/dev/null || true; exit $_rc' INT TERM

# ── State tracking ─────────────────────────────────────────────
BASELINE_RUNS=0
OPTIMIZED_RUNS=0
FAILED_RUNS=0

run_one() {
    local variant="$1"   # baseline | optimized
    local run_num="$2"

    step "Run ${run_num}/${TOTAL_RUNS} — ${variant^^}"

    if "$SCRIPT_DIR/fl.sh" train "$variant"; then
        info "  ✓ ${variant} run ${run_num} complete"
        if [ "$variant" = "baseline" ]; then
            (( BASELINE_RUNS += 1 ))
        else
            (( OPTIMIZED_RUNS += 1 ))
        fi
    else
        warn "  ✗ ${variant} run ${run_num} FAILED (exit $?) — stopping pipeline"
        (( FAILED_RUNS++ ))
        return 1
    fi

    # Brief pause so ports and Pi processes fully release before the next run
    sleep 5
}

# ── Pipeline header ────────────────────────────────────────────
step "FL PIPELINE START"
info "  Runs per mode   : ${TOTAL_RUNS}"
info "  Rounds per run  : ${NUM_ROUNDS:-10}"
info "  Output layout   : outputs/baseline_<ts>/   outputs/optimized_<ts>/"
if [ -n "${RUN_STEPS:-}" ]; then
    info "  Mode            : SPECIFIC STEPS  (RUN_STEPS=${RUN_STEPS})"
else
    info "  Mode            : RANGE  (START_STEP=${START_STEP} → MAX_STEP=$((TOTAL_RUNS * 2)))"
fi
info ""

PIPELINE_START=$(date +%s)

_run_step() {
    local step_idx="$1"
    local run_num=$(( (step_idx + 1) / 2 ))

    # First half (steps 1..HALF) → Pinata account_1
    # Second half (steps HALF+1..MAX_STEP) → Pinata account_2
    if [ "$step_idx" -le "$HALF" ]; then
        export PINATA_ACCOUNT=1
    else
        export PINATA_ACCOUNT=2
    fi
    info "  [PINATA] step ${step_idx}: PINATA_ACCOUNT=${PINATA_ACCOUNT}"

    if (( step_idx % 2 == 1 )); then
        run_one baseline "$run_num"
    else
        run_one optimized "$run_num"
    fi
}

# ── Main loop ─────────────────────────────────────────────────
MAX_STEP=$((TOTAL_RUNS * 2))

if [ -n "${RUN_STEPS:-}" ]; then
    # Run only the explicitly listed step indices
    for step_idx in $RUN_STEPS; do
        if [ "$step_idx" -lt 1 ] || [ "$step_idx" -gt "$MAX_STEP" ]; then
            warn "Skipping invalid step ${step_idx} (valid range: 1..${MAX_STEP})"
            continue
        fi
        _run_step "$step_idx"
    done
else
    # Run the full range from START_STEP
    if [ "$START_STEP" -lt 1 ] || [ "$START_STEP" -gt "$MAX_STEP" ]; then
        warn "Invalid START_STEP=${START_STEP}. Expected 1..${MAX_STEP}."
        exit 1
    fi
    for ((step_idx=START_STEP; step_idx<=MAX_STEP; step_idx++)); do
        _run_step "$step_idx"
    done
fi

# ── Summary ───────────────────────────────────────────────────
ELAPSED=$(( $(date +%s) - PIPELINE_START ))

step "PIPELINE COMPLETE"
info "  Baseline runs   : ${BASELINE_RUNS}/${TOTAL_RUNS}"
info "  Optimized runs  : ${OPTIMIZED_RUNS}/${TOTAL_RUNS}"
[ "$FAILED_RUNS" -gt 0 ] && warn "  Failed runs     : ${FAILED_RUNS}" || true
info "  Elapsed         : $(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m $(( ELAPSED % 60 ))s"
info ""
info "  Baseline results:"
ls -d "${SCRIPT_DIR}/outputs/baseline_"* 2>/dev/null \
    | while read -r d; do info "    $d"; done \
    || info "    (none found)"
info "  Optimized results:"
ls -d "${SCRIPT_DIR}/outputs/optimized_"* 2>/dev/null \
    | while read -r d; do info "    $d"; done \
    || info "    (none found)"
