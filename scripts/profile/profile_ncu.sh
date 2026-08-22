#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MODEL_PATH="${MODEL_PATH:-models/qwen3-0.6B}"
BINARY="${BINARY:-build/src/ccinfer-server}"
PORT="${PORT:-8081}"
OUT_DIR="${OUT_DIR:-docs/profiling/baseline/ncu}"
RUN_NAME="${RUN_NAME:-decode_attention_p32_o16_c1}"
PROMPT_LEN="${PROMPT_LEN:-32}"
MAX_TOKENS="${MAX_TOKENS:-16}"
CONCURRENCY="${CONCURRENCY:-1}"
REQUESTS="${REQUESTS:-1}"
KERNEL_NAME="${KERNEL_NAME:-regex:.*decode_attention.*}"
LAUNCH_COUNT="${LAUNCH_COUNT:-10}"
LAUNCH_SKIP="${LAUNCH_SKIP:-0}"
NCU_SET="${NCU_SET:-basic}"

mkdir -p "${OUT_DIR}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate llm-infer
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill -TERM "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "[ncu] output: ${OUT_DIR}/${RUN_NAME}.ncu-rep"
ncu \
  --target-processes all \
  --set "${NCU_SET}" \
  --kernel-name "${KERNEL_NAME}" \
  --launch-skip "${LAUNCH_SKIP}" \
  --launch-count "${LAUNCH_COUNT}" \
  --force-overwrite \
  --export "${OUT_DIR}/${RUN_NAME}" \
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

ncu --import "${OUT_DIR}/${RUN_NAME}.ncu-rep" --page raw > "${OUT_DIR}/${RUN_NAME}_raw.txt" || true
if grep -q "ERR_NVGPUCTRPERM" "${OUT_DIR}/${RUN_NAME}_raw.txt" 2>/dev/null; then
  cat >&2 <<'EOF'
[ncu] GPU performance counters are blocked (ERR_NVGPUCTRPERM).
[ncu] On this WSL2 setup, enable counters on the Windows host:
[ncu]   NVIDIA Control Panel (Admin) -> Desktop -> Enable Developer Settings
[ncu]   Developer -> Manage GPU Performance Counters
[ncu]   Allow access to the GPU performance counter to all users
[ncu] Then run: wsl --shutdown
[ncu] See docs/profiling/enable_ncu_counters_wsl2.md
EOF
fi
echo "[ncu] done"
