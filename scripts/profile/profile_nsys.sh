#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MODEL_PATH="${MODEL_PATH:-models/qwen3-0.6B}"
BINARY="${BINARY:-build/src/ccinfer-server}"
PORT="${PORT:-8080}"
OUT_DIR="${OUT_DIR:-docs/profiling/baseline/nsys}"
RUN_NAME="${RUN_NAME:-p32_o16_c1}"
PROMPT_LEN="${PROMPT_LEN:-32}"
MAX_TOKENS="${MAX_TOKENS:-16}"
CONCURRENCY="${CONCURRENCY:-1}"
REQUESTS="${REQUESTS:-2}"

mkdir -p "${OUT_DIR}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate llm-infer
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill -INT "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "[nsys] output: ${OUT_DIR}/${RUN_NAME}.nsys-rep"
nsys profile \
  --trace=cuda,cublas,nvtx,osrt \
  --sample=cpu \
  --cpuctxsw=process-tree \
  --force-overwrite=true \
  --output="${OUT_DIR}/${RUN_NAME}" \
  "${BINARY}" --port "${PORT}" --model-path "${MODEL_PATH}" &
SERVER_PID=$!

python3 - <<PY
import time
import requests

url = "http://127.0.0.1:${PORT}/health"
deadline = time.time() + 180
while time.time() < deadline:
    try:
        r = requests.get(url, timeout=2)
        if r.status_code == 200:
            raise SystemExit(0)
    except Exception:
        pass
    time.sleep(0.5)
raise SystemExit("server did not become healthy")
PY

python3 scripts/profile/profile_client.py \
  --base-url "http://127.0.0.1:${PORT}" \
  --prompt-len "${PROMPT_LEN}" \
  --max-tokens "${MAX_TOKENS}" \
  --concurrency "${CONCURRENCY}" \
  --requests "${REQUESTS}" | tee "${OUT_DIR}/${RUN_NAME}_client.json"

cleanup
trap - EXIT

nsys stats --force-export=true --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,osrt_sum \
  "${OUT_DIR}/${RUN_NAME}.nsys-rep" > "${OUT_DIR}/${RUN_NAME}_stats.txt" || true

echo "[nsys] done"
