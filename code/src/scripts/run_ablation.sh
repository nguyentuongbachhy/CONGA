#!/usr/bin/env bash
# Full sequential ablation study for CONGA (architecture components).
# Usage (from code/src/):
#   bash scripts/run_ablation.sh [options]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$SRC/ablation_logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/master.log"

PYTHON="uv run python"

TORCH_LIB=$(uv run python -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')" 2>/dev/null)
if [ -n "$TORCH_LIB" ]; then
    export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
fi

A3_CKPT=""
DATASET="ml-1m"
MAXLEN="300"
MEM_MAXLEN="600"
BATCH_SIZE="128"
LR="0.001"
HIDDEN_UNITS="64"
NUM_BLOCKS="2"
NUM_HEADS="1"       # Fix 1: was 4, SASRec/CONGA uses 1
DROPOUT_RATE="0.2"
LOSS_TYPE="ce"
NUM_NEGATIVES="1"   # CE path ignores negatives, keep at 1 to avoid 0-dim tensor
NUM_EPOCHS="600"
EVAL_EVERY="5"
PATIENCE="20"
SEED="42"

while [[ $# -gt 0 ]]; do
    case $1 in
        --a3_ckpt)       A3_CKPT="$2";       shift 2 ;;
        --dataset)       DATASET="$2";       shift 2 ;;
        --maxlen)        MAXLEN="$2";        shift 2 ;;
        --mem_maxlen)    MEM_MAXLEN="$2";    shift 2 ;;
        --batch_size)    BATCH_SIZE="$2";    shift 2 ;;
        --lr)            LR="$2";            shift 2 ;;
        --hidden_units)  HIDDEN_UNITS="$2";  shift 2 ;;
        --num_blocks)    NUM_BLOCKS="$2";    shift 2 ;;
        --num_heads)     NUM_HEADS="$2";     shift 2 ;;
        --dropout_rate)  DROPOUT_RATE="$2";  shift 2 ;;
        --loss_type)     LOSS_TYPE="$2";     shift 2 ;;
        --num_negatives) NUM_NEGATIVES="$2"; shift 2 ;;
        --num_epochs)    NUM_EPOCHS="$2";    shift 2 ;;
        --eval_every)    EVAL_EVERY="$2";    shift 2 ;;
        --patience)      PATIENCE="$2";      shift 2 ;;
        --seed)          SEED="$2";          shift 2 ;;
        *) echo "Unknown arg: $1" >&2; shift ;;
    esac
done

C="--dataset $DATASET --batch_size $BATCH_SIZE --maxlen $MAXLEN
 --num_blocks $NUM_BLOCKS --num_heads $NUM_HEADS --hidden_units $HIDDEN_UNITS
 --dropout_rate $DROPOUT_RATE --seed $SEED --num_negatives $NUM_NEGATIVES
 --num_workers 2 --model_type conga
 --eval_every $EVAL_EVERY --patience $PATIENCE --lr $LR
 --grad_clip 1.0 --mask_ratio 0.15"
R="--loss_type $LOSS_TYPE --cosine_anneal --warmup_ratio 0.05"

_run() {
    local id=$1; shift
    local out="$LOG_DIR/abl_${id}.log"   # stdout captured here
    local dir="$SRC/${DATASET}_abl_${id}"

    # Skip only if previous run completed with a meaningful result (N@10 > 0.01).
    # This prevents stale logs from a failed/diverged run from blocking re-runs.
    if [ -f "$out" ]; then
        local prev_n10
        prev_n10=$(grep "^Best Test" "$out" 2>/dev/null | tail -1 | grep -oP "N@10: \K[\d.]+")
        if [ -n "$prev_n10" ] && awk "BEGIN{exit !($prev_n10 > 0.01)}"; then
            echo "=== [$(date '+%H:%M:%S')] SKIP $id (already done, N@10=$prev_n10) ===" | tee -a "$MASTER_LOG"
            grep "^Best Test" "$out" | tail -1 | tee -a "$MASTER_LOG"
            return 0
        elif [ -n "$prev_n10" ]; then
            echo "=== [$(date '+%H:%M:%S')] RERUN $id (stale result N@10=$prev_n10 < 0.01, re-running) ===" | tee -a "$MASTER_LOG"
        fi
    fi

    echo "" | tee -a "$MASTER_LOG"
    echo "=== [$(date '+%H:%M:%S')] START $id ===" | tee -a "$MASTER_LOG"
    cd "$SRC" && $PYTHON main.py $C "$@" --train_dir "abl_${id}" 2>&1 | tee "$out"
    grep "^Best Test" "$out" | tee -a "$MASTER_LOG"
    echo "=== [$(date '+%H:%M:%S')] DONE $id ===" | tee -a "$MASTER_LOG"
}

echo "=== [$(date '+%H:%M:%S')] ABLATION START ===" | tee "$MASTER_LOG"
echo "Config: dataset=$DATASET heads=$NUM_HEADS loss=$LOSS_TYPE lr=$LR epochs=$NUM_EPOCHS" | tee -a "$MASTER_LOG"
cd "$SRC" && $PYTHON -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>/dev/null | tee -a "$MASTER_LOG"

# Run A2 first (A3 depends on its checkpoint), then A3, then A1/A0.

# A2: + KromHC + SwiGLU (CONGA without TITANS) — RoPE & num_streams adaptive
if [ -n "$A3_CKPT" ]; then
    echo "=== [$(date '+%H:%M:%S')] SKIP A2 (using provided checkpoint: $A3_CKPT) ===" | tee -a "$MASTER_LOG"
    A2_BEST="$A3_CKPT"
else
    _run A2 $R --num_epochs $NUM_EPOCHS
    A2_BEST="$SRC/${DATASET}_abl_A2/SASRec.best.pth"
fi

# A3: Full CONGA (A2 + TITANS, phase-2 fine-tune)
_run A3 $R \
    --use_nested_learning --mem_start_epoch 1 --mem_maxlen $MEM_MAXLEN \
    --titans_d_mem 128 --titans_base_lr_scale 0.1 \
    --titans_mem_lr_scale 0.5 --titans_mem_wd 0.01 \
    --state_dict_path "$A2_BEST" \
    --num_epochs 80

# A1: + KromHC — RoPE & num_streams adaptive
_run A1 --no_swiglu $R --num_epochs $NUM_EPOCHS

# A0: CONGA base (no KromHC, no SwiGLU) — RoPE & num_streams adaptive
_run A0 --no_mhc --no_swiglu $R --num_epochs $NUM_EPOCHS

# ── Summary ──
echo "" | tee -a "$MASTER_LOG"
printf '=%.0s' {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50} | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "ABLATION SUMMARY ($DATASET)" | tee -a "$MASTER_LOG"
printf '=%.0s' {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50} | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
printf "%-4s %-40s %8s %8s %8s %8s\n" "ID" "Description" "N@5" "H@5" "N@10" "H@10" | tee -a "$MASTER_LOG"
echo "------------------------------------------------------------------------" | tee -a "$MASTER_LOG"

declare -A DESCS
DESCS[A0]="CONGA base (no KromHC/SwiGLU, adaptive RoPE)"
DESCS[A1]="A0 + KromHC"
DESCS[A2]="A1 + SwiGLU  (CONGA w/o TITANS)"
DESCS[A3]="Full CONGA   (A2 + TITANS)"

for id in A0 A1 A2 A3; do
    out="$LOG_DIR/abl_${id}.log"
    line=$(grep "^Best Test" "$out" 2>/dev/null | tail -1)
    if [ -n "$line" ]; then
        n5=$(echo  "$line" | grep -oP "N@5: \K[\d.]+")
        h5=$(echo  "$line" | grep -oP "H@5: \K[\d.]+")
        n10=$(echo "$line" | grep -oP "N@10: \K[\d.]+")
        h10=$(echo "$line" | grep -oP "H@10: \K[\d.]+")
        printf "%-4s %-40s %8s %8s %8s %8s\n" "$id" "${DESCS[$id]}" "$n5" "$h5" "$n10" "$h10" | tee -a "$MASTER_LOG"
    else
        printf "%-4s %-40s %8s\n" "$id" "${DESCS[$id]}" "N/A" | tee -a "$MASTER_LOG"
    fi
done

echo "========================================================================" | tee -a "$MASTER_LOG"
echo "[$(date '+%H:%M:%S')] ALL DONE. Full log: $MASTER_LOG" | tee -a "$MASTER_LOG"