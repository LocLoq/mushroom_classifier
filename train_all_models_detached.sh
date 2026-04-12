#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "${1:-}" != "--child" ]]; then
  mkdir -p logs
  timestamp="$(date +%Y%m%d_%H%M%S)"
  log_file="logs/train_eval_plot_${timestamp}.log"
  pid_file="logs/train_eval_plot_${timestamp}.pid"

  nohup bash "$0" --child >"$log_file" 2>&1 &
  child_pid=$!
  echo "$child_pid" >"$pid_file"

  echo "Detached pipeline started"
  echo "PID: $child_pid"
  echo "PID file: $pid_file"
  echo "Log file: $log_file"
  exit 0
fi
shift

if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
elif [[ -x "$SCRIPT_DIR/.venv/Scripts/python.exe" ]]; then
  PYTHON_BIN="$SCRIPT_DIR/.venv/Scripts/python.exe"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

DISCORD_WEBHOOK_URL="${DISCORD_WEBHOOK_URL:-}"
DISCORD_NOTIFY_EVERY_EPOCHS="${DISCORD_NOTIFY_EVERY_EPOCHS:-1}"

echo "Using python: $PYTHON_BIN"
"$PYTHON_BIN" --version

notify_discord() {
  local message="$1"

  if [[ -z "$DISCORD_WEBHOOK_URL" ]]; then
    return 0
  fi

  DISCORD_WEBHOOK_URL="$DISCORD_WEBHOOK_URL" DISCORD_MESSAGE="$message" "$PYTHON_BIN" - <<'PY'
import json
import os
import urllib.request

webhook = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
message = os.environ.get("DISCORD_MESSAGE", "").strip()

if webhook and message:
    if len(message) > 1900:
        message = message[:1897] + "..."
    payload = json.dumps({"content": message}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        webhook,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(req, timeout=10).read()
PY
}

current_stage="init"
on_error() {
  local exit_code=$?
  notify_discord "[Pipeline Failed] stage=${current_stage} | exit_code=${exit_code}"
  echo "Pipeline failed at stage=${current_stage}, exit_code=${exit_code}"
  exit "$exit_code"
}
trap on_error ERR

TRAIN_DISCORD_ARGS=()
if [[ -n "$DISCORD_WEBHOOK_URL" ]]; then
  TRAIN_DISCORD_ARGS=(
    --discord-webhook-url "$DISCORD_WEBHOOK_URL"
    --discord-notify-every-epochs "$DISCORD_NOTIFY_EVERY_EPOCHS"
  )
fi

notify_discord "[Pipeline Start] Train/Eval/Plot for models: efficientnet_b0, resnet18, vgg16"

TRAIN_COMMON_ARGS=(
  --data-dir mushrooms_dataset
  --train-subdir train
  --val-subdir val
  --epochs 100
  --min-epochs 20
  --early-stop-patience 14
  --batch-size 192
  --num-workers 12
  --lr 0.0003
  --backbone-lr 0.0001
  --weight-decay 0.0001
  --label-smoothing 0.05
  --unfreeze-epoch 4
  --unfreeze-blocks 3
  --scheduler cosine
  --min-lr 1e-6
  --save-every-batches 200
  --no-append-metrics
)

MODELS=(efficientnet_b0 resnet18 vgg16)

for model_name in "${MODELS[@]}"; do
  echo "========================================"
  echo "[TRAIN] model=${model_name}"
  echo "========================================"
  current_stage="train:${model_name}"
  notify_discord "[Stage Start] ${current_stage}"

  "$PYTHON_BIN" train.py \
    --model "$model_name" \
    "${TRAIN_COMMON_ARGS[@]}" \
    "${TRAIN_DISCORD_ARGS[@]}" \
    --model-out "nammushroom_${model_name}.pth" \
    --best-model-out "best_nammushroom_${model_name}.pth" \
    --checkpoint-path "checkpoints/latest_checkpoint_${model_name}.pth" \
    --best-checkpoint-path "checkpoints/best_checkpoint_${model_name}.pth" \
    --metrics-csv "training_metrics_${model_name}.csv" \
    --batch-metrics-csv "training_batch_metrics_${model_name}.csv" \
    --confusion-dir "confusion_matrices_${model_name}" \
    --report-path "training_report_${model_name}.json"
  notify_discord "[Stage Done] ${current_stage}"

  echo "========================================"
  echo "[EVAL] model=${model_name}"
  echo "========================================"
  current_stage="eval:${model_name}"
  notify_discord "[Stage Start] ${current_stage}"

  "$PYTHON_BIN" evaluate_test.py \
    --model "$model_name" \
    --data-dir mushrooms_dataset \
    --train-subdir train \
    --test-subdir test \
    --model-path "best_nammushroom_${model_name}.pth" \
    --batch-size 768 \
    --num-workers 12 \
    --metrics-csv "test_metrics_${model_name}.csv" \
    --confusion-dir "confusion_matrices_test_${model_name}"
  notify_discord "[Stage Done] ${current_stage}"

  echo "========================================"
  echo "[PLOT SINGLE] model=${model_name}"
  echo "========================================"
  current_stage="plot-single:${model_name}"
  notify_discord "[Stage Start] ${current_stage}"

  "$PYTHON_BIN" plot_metrics.py \
    --metrics-csv "training_metrics_${model_name}.csv" \
    --batch-metrics-csv "training_batch_metrics_${model_name}.csv" \
    --test-metrics-csv "test_metrics_${model_name}.csv" \
    --confusion-dir "confusion_matrices_${model_name}" \
    --output-dir "plots/${model_name}" \
    --run-id latest \
    --max-heatmap-classes 80 \
    --max-batch-points 20000
  notify_discord "[Stage Done] ${current_stage}"

done

echo "========================================"
echo "[PLOT COMPARE] efficientnet_b0 vs resnet18 vs vgg16"
echo "========================================"
current_stage="plot-compare"
notify_discord "[Stage Start] ${current_stage}"

"$PYTHON_BIN" plot_metrics.py \
  --output-dir "plots/compare_all" \
  --max-heatmap-classes 80 \
  --max-batch-points 20000 \
  --skip-confusion-heatmaps \
  --compare-run-id latest \
  --compare-metrics "efficientnet_b0=training_metrics_efficientnet_b0.csv" \
  --compare-metrics "resnet18=training_metrics_resnet18.csv" \
  --compare-metrics "vgg16=training_metrics_vgg16.csv" \
  --compare-batch-metrics "efficientnet_b0=training_batch_metrics_efficientnet_b0.csv" \
  --compare-batch-metrics "resnet18=training_batch_metrics_resnet18.csv" \
  --compare-batch-metrics "vgg16=training_batch_metrics_vgg16.csv"
notify_discord "[Stage Done] ${current_stage}"

echo "Pipeline finished successfully."
notify_discord "[Pipeline Done] All models completed successfully. Output: plots/compare_all"
