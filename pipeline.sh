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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOTAL_RUNS="${TOTAL_RUNS:-10}"

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
        [ "$variant" = "baseline" ] && (( BASELINE_RUNS++ )) || (( OPTIMIZED_RUNS++ ))
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
info "  Total runs      : $((TOTAL_RUNS * 2))  (baseline → optimized, alternating)"
info "  Rounds per run  : ${NUM_ROUNDS:-10}"
info "  Total rounds    : $((TOTAL_RUNS * ${NUM_ROUNDS:-10})) baseline + $((TOTAL_RUNS * ${NUM_ROUNDS:-10})) optimized"
info "  Output layout   : outputs/baseline_<ts>/   outputs/optimized_<ts>/"
info ""

PIPELINE_START=$(date +%s)

# ── Main loop: baseline → optimized × TOTAL_RUNS ──────────────
for ((i=1; i<=TOTAL_RUNS; i++)); do
    run_one baseline  "$i"
    run_one optimized "$i"
done

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
