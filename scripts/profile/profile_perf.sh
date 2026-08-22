#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if ! command -v perf >/dev/null 2>&1; then
  echo "perf not found in PATH. Install linux-tools/perf or run on a host with perf available." >&2
  exit 127
fi

find_tool() {
  local name="$1"
  local fallback="$2"
  if command -v "${name}" >/dev/null 2>&1; then
    command -v "${name}"
  elif [[ -x "${fallback}" ]]; then
    printf '%s\n' "${fallback}"
  else
    return 1
  fi
}

FLAMEGRAPH_PL="$(find_tool flamegraph.pl /home/qic7/FlameGraph/flamegraph.pl || true)"
STACKCOLLAPSE_PERF_PL="$(find_tool stackcollapse-perf.pl /home/qic7/FlameGraph/stackcollapse-perf.pl || true)"

MODEL_PATH="${MODEL_PATH:-models/qwen3-0.6B}"
BINARY="${BINARY:-build/src/ccinfer-server}"
PORT="${PORT:-8082}"
OUT_DIR="${OUT_DIR:-docs/profiling/baseline/perf}"
RUN_NAME="${RUN_NAME:-p32_o16_c1}"
PROMPT_LEN="${PROMPT_LEN:-32}"
MAX_TOKENS="${MAX_TOKENS:-16}"
CONCURRENCY="${CONCURRENCY:-1}"
REQUESTS="${REQUESTS:-4}"

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

perf record -F 99 -g -o "${OUT_DIR}/${RUN_NAME}.perf.data" -- \
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

perf report --stdio -i "${OUT_DIR}/${RUN_NAME}.perf.data" > "${OUT_DIR}/${RUN_NAME}_report.txt" || true
if [[ -n "${FLAMEGRAPH_PL}" && -n "${STACKCOLLAPSE_PERF_PL}" ]]; then
  perf script -i "${OUT_DIR}/${RUN_NAME}.perf.data" \
    | "${STACKCOLLAPSE_PERF_PL}" > "${OUT_DIR}/${RUN_NAME}_folded.txt" || true
  "${FLAMEGRAPH_PL}" "${OUT_DIR}/${RUN_NAME}_folded.txt" \
    > "${OUT_DIR}/${RUN_NAME}_flamegraph.svg" || true
else
  echo "[perf] FlameGraph scripts not found; skipped SVG generation." >&2
fi
echo "[perf] done"
