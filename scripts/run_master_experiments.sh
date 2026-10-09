#!/usr/bin/env bash
#
# Master's thesis experiment driver.
#
#   Research question: does transition-based intrinsic reward improve
#   exploration compared with the original BiGAN-based state novelty used in
#   Adventurer?
#
#   Experiment A (baseline)  --method state       BiGAN state novelty B(s)
#   Experiment B (proposed)  --method transition  BiGAN transition novelty
#                            N_T = ||f(E(s_t),a_t) - E(s_{t+1})||_2
#
# Both conditions share the environment, preprocessing, PPO architecture and
# hyperparameters, rollout geometry, training budget, and seeds. The intrinsic
# exploration signal is the only intended difference. No hyperparameter is tuned
# per method.
#
# The legacy EME/metric bonus and the legacy transition novelty
# (--novelty transition) are NOT part of this experiment and are never launched
# by this script.
#
# Environment: the default is the Noisy-TV maze of "Large-Scale Study of
# Curiosity-Driven Learning" (the Unity project luchris429/noisy-tv-env),
# reconstructed in pure python in environments/noisy_tv_maze.py: 17 rooms and
# 18 corridors, 84x84 RGB first-person frames, six discrete actions, a sparse
# +1 reward 2.5 units from the goal sphere, and the stochastic television the
# paper uses to expose the noisy-TV failure mode.
#
#   ENV_NAME=NoisyTVMaze-v0    the reconstruction (no Unity, no ROM, CPU-friendly)
#   ENV_NAME=NoisyTVUnity-v0   the original Unity build, when the executable
#                              and the upstream `unityagents` client are present
#   ENV_NAME=ALE/MontezumaRevenge-v5   the Atari benchmark used elsewhere in the thesis
#
# Usage:
#   bash scripts/run_master_experiments.sh
#   DEVICE=cpu NUM_ENVS=16 TOTAL_STEPS=1024000 SEEDS="0 1 2" \
#       bash scripts/run_master_experiments.sh                      # quick scale
#   ENV_NAME=ALE/MontezumaRevenge-v5 DEVICE=cuda NUM_ENVS=96 \
#       TOTAL_STEPS=12288000 SEEDS="0 1 2" bash scripts/run_master_experiments.sh
#
# Configurable variables (defaults in brackets):
#   ENV_NAME              [NoisyTVMaze-v0]  or any registered Gymnasium ID,
#                                           e.g. NoisyTVUnity-v0,
#                                           ALE/MontezumaRevenge-v5,
#                                           CartPole-v1
#   DEVICE                [auto]     auto|cpu|cuda  (auto = cuda if available)
#   NUM_ENVS              [96]
#   ROLLOUT_STEPS         [128]
#   MINIBATCH_SIZE        [32]
#   TOTAL_STEPS           [12288000]
#   BIGAN_BATCH_SIZE      [64]
#   SEEDS                 [0 1 2]
#   METHODS               [state transition]
#   OUTPUT_DIR            [results/master]
#   PYTHON                [python3]  or set VENV=/path/to/venv
#   PLOT_INTERVAL_UPDATES [50]       logging frequency only, identical for both
#   DISABLE_TENSORBOARD   [0]
#   DISABLE_PLOTS         [0]
#   OVERWRITE             [0]        1 = delete an existing run directory first
#   EXTRA_ARGS            []         passed verbatim to every run (both methods)
#
# The script never retries a failed run: a failure stops everything so that
# reproducibility problems stay visible.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${ROOT_DIR}"

ENV_NAME="${ENV_NAME:-NoisyTVMaze-v0}"
DEVICE="${DEVICE:-auto}"
NUM_ENVS="${NUM_ENVS:-96}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-128}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-32}"
TOTAL_STEPS="${TOTAL_STEPS:-12288000}"
BIGAN_BATCH_SIZE="${BIGAN_BATCH_SIZE:-64}"
SEEDS="${SEEDS:-0 1 2}"
METHODS="${METHODS:-state transition}"
OUTPUT_DIR="${OUTPUT_DIR:-results/master}"
PLOT_INTERVAL_UPDATES="${PLOT_INTERVAL_UPDATES:-50}"
DISABLE_TENSORBOARD="${DISABLE_TENSORBOARD:-0}"
DISABLE_PLOTS="${DISABLE_PLOTS:-0}"
OVERWRITE="${OVERWRITE:-0}"

if [[ -n "${VENV:-}" && -x "${VENV}/bin/python" ]]; then
    PYTHON="${VENV}/bin/python"
fi
PYTHON="${PYTHON:-python3}"
read -r -a EXTRA_ARGS_ARRAY <<< "${EXTRA_ARGS:-}"

if ! command -v "${PYTHON}" >/dev/null 2>&1 && [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: python interpreter not found: ${PYTHON}" >&2
    exit 1
fi

echo "============================================================"
echo " Master's thesis experiment: BiGAN state vs. transition novelty"
echo "============================================================"
echo "repository root : ${ROOT_DIR}"
GIT_COMMIT="$(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
if [[ -n "$(git -C "${ROOT_DIR}" status --porcelain 2>/dev/null)" ]]; then
    GIT_DIRTY="yes (uncommitted changes present -- record them before publishing results)"
else
    GIT_DIRTY="no"
fi
echo "git commit      : ${GIT_COMMIT}"
echo "git branch      : ${GIT_BRANCH}"
echo "working tree dirty: ${GIT_DIRTY}"
echo "started (UTC)   : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "environment     : ${ENV_NAME}"
echo "device          : ${DEVICE}"
echo "num envs        : ${NUM_ENVS}"
echo "rollout steps   : ${ROLLOUT_STEPS}"
echo "minibatch size  : ${MINIBATCH_SIZE}"
echo "total steps     : ${TOTAL_STEPS}"
echo "BiGAN batch     : ${BIGAN_BATCH_SIZE}"
echo "seeds           : ${SEEDS}"
echo "methods         : ${METHODS}"
echo "output dir      : ${OUTPUT_DIR}"
echo "python          : ${PYTHON}"
echo "------------------------------------------------------------"

# 1. Dependency check -------------------------------------------------------
if ! "${PYTHON}" - <<'PY'
import sys

missing = []
for module in ("torch", "numpy", "gymnasium", "matplotlib"):
    try:
        __import__(module)
    except ImportError:
        missing.append(module)
if missing:
    print("missing python packages: " + ", ".join(missing), file=sys.stderr)
    sys.exit(1)
import numpy, torch, gymnasium
print(f"python     {sys.version.split()[0]}")
print(f"torch      {torch.__version__}")
print(f"gymnasium  {gymnasium.__version__}")
print(f"numpy      {numpy.__version__}")
try:
    import ale_py
    print(f"ale-py     {ale_py.__version__}")
except ImportError:
    print("ale-py     not installed (required for ALE/* environments)")
PY
then
    echo "ERROR: dependency check failed. Install with:" >&2
    echo "  ${PYTHON} -m pip install -r requirements-master.txt" >&2
    exit 1
fi

if [[ "${ENV_NAME}" == ALE/* ]]; then
    if ! "${PYTHON}" -c "import ale_py" >/dev/null 2>&1; then
        echo "ERROR: ${ENV_NAME} requires ale-py (see requirements-master.txt)" >&2
        exit 1
    fi
fi

# The Noisy-TV environments live in this repository, so check up front that the
# requested one can actually be created instead of failing on the first run.
if [[ "${ENV_NAME}" == NoisyTV* ]]; then
    if ! "${PYTHON}" -c "
from environments.noisy_tv_unity import register_environment, unity_available

register_environment()
if '${ENV_NAME}' == 'NoisyTVUnity-v0' and not unity_available():
    raise SystemExit('the original Unity build or its client is missing')
" >/dev/null 2>&1; then
        echo "ERROR: ${ENV_NAME} is not available." >&2
        echo "       NoisyTVMaze-v0 is the pure-python reconstruction and needs gymnasium only." >&2
        echo "       NoisyTVUnity-v0 additionally needs the Unity player executable and the" >&2
        echo "       upstream unityagents client; see environments/noisy_tv_unity.py for the" >&2
        echo "       download and install steps, or use make_noisy_tv_environment('auto')." >&2
        exit 1
    fi
fi

# 2. Schedule consistency ---------------------------------------------------
SAMPLES_PER_UPDATE=$(( NUM_ENVS * ROLLOUT_STEPS ))
if (( SAMPLES_PER_UPDATE <= 0 )); then
    echo "ERROR: NUM_ENVS * ROLLOUT_STEPS must be positive" >&2
    exit 1
fi
if (( TOTAL_STEPS % SAMPLES_PER_UPDATE != 0 )); then
    echo "ERROR: TOTAL_STEPS (${TOTAL_STEPS}) must be divisible by NUM_ENVS * ROLLOUT_STEPS (${SAMPLES_PER_UPDATE})" >&2
    exit 1
fi
if (( ROLLOUT_STEPS % MINIBATCH_SIZE != 0 )); then
    echo "ERROR: ROLLOUT_STEPS (${ROLLOUT_STEPS}) must be divisible by MINIBATCH_SIZE (${MINIBATCH_SIZE})" >&2
    exit 1
fi
echo "updates per run : $(( TOTAL_STEPS / SAMPLES_PER_UPDATE ))"
echo "------------------------------------------------------------"

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS=(
    --env "${ENV_NAME}"
    --num-parallel-envs "${NUM_ENVS}"
    --rollout-steps "${ROLLOUT_STEPS}"
    --minibatch-size "${MINIBATCH_SIZE}"
    --total-environment-steps "${TOTAL_STEPS}"
    --bigan-batch-size "${BIGAN_BATCH_SIZE}"
    --device "${DEVICE}"
    --plot-interval-updates "${PLOT_INTERVAL_UPDATES}"
)
if [[ "${DISABLE_TENSORBOARD}" == "1" ]]; then
    COMMON_ARGS+=(--disable-tensorboard)
fi
if [[ "${DISABLE_PLOTS}" == "1" ]]; then
    COMMON_ARGS+=(--disable-plots)
fi

TOTAL_RUNS=0
for _ in ${METHODS}; do
    for _ in ${SEEDS}; do
        TOTAL_RUNS=$(( TOTAL_RUNS + 1 ))
    done
done

SUCCESSFUL_RUNS=()
FAILED_RUN=""
RUN_INDEX=0

for METHOD in ${METHODS}; do
    if [[ "${METHOD}" != "state" && "${METHOD}" != "transition" ]]; then
        echo "ERROR: METHODS may only contain 'state' and 'transition', got '${METHOD}'" >&2
        exit 1
    fi
    for SEED in ${SEEDS}; do
        RUN_INDEX=$(( RUN_INDEX + 1 ))
        RUN_DIR="${OUTPUT_DIR}/${METHOD}/seed_${SEED}"
        LOG_FILE="${RUN_DIR}/training.log"

        echo ""
        echo "============================================================"
        echo "[${RUN_INDEX}/${TOTAL_RUNS}] method=${METHOD} seed=${SEED}"
        echo "output: ${RUN_DIR}"
        echo "log   : ${LOG_FILE}"
        echo "============================================================"

        if [[ -e "${RUN_DIR}/metrics.jsonl" ]]; then
            if [[ "${OVERWRITE}" == "1" ]]; then
                echo "OVERWRITE=1: removing existing run directory ${RUN_DIR}"
                rm -rf "${RUN_DIR}"
            else
                echo "ERROR: ${RUN_DIR}/metrics.jsonl already exists." >&2
                echo "       The logger appends, so re-running into the same directory would mix" >&2
                echo "       two experiments. Choose a fresh OUTPUT_DIR or set OVERWRITE=1." >&2
                exit 1
            fi
        fi
        mkdir -p "${RUN_DIR}"

        START_SECONDS="${SECONDS}"
        # -u: unbuffered, so training.log stays useful while the run is live.
        # stdout and stderr are merged into training.log and echoed to the
        # console; `set -o pipefail` makes a python failure fail the pipeline.
        if ! "${PYTHON}" -u main.py \
            --method "${METHOD}" \
            --seed "${SEED}" \
            --output-directory "${RUN_DIR}" \
            --plot-directory "${RUN_DIR}/plt" \
            --tensorboard-directory "${RUN_DIR}/tensorboard" \
            "${COMMON_ARGS[@]}" \
            ${EXTRA_ARGS_ARRAY[@]+"${EXTRA_ARGS_ARRAY[@]}"} 2>&1 | tee -a "${LOG_FILE}"; then
            FAILED_RUN="${METHOD}/seed_${SEED}"
            echo "" >&2
            echo "ERROR: run failed: ${FAILED_RUN} (see ${LOG_FILE})" >&2
            echo "       Not retrying: a failed run must stay visible." >&2
            exit 1
        fi
        ELAPSED=$(( SECONDS - START_SECONDS ))
        echo "[${RUN_INDEX}/${TOTAL_RUNS}] finished ${METHOD}/seed_${SEED} in ${ELAPSED}s"
        SUCCESSFUL_RUNS+=("${METHOD}/seed_${SEED}=${ELAPSED}s")
    done
done

echo ""
echo "============================================================"
echo " Aggregating results"
echo "============================================================"
"${PYTHON}" scripts/summarize_master_results.py --results-dir "${OUTPUT_DIR}" --methods ${METHODS}
if ! "${PYTHON}" scripts/plot_master_results.py --results-dir "${OUTPUT_DIR}" --methods ${METHODS}; then
    echo "WARNING: figure generation failed; the CSV summaries above are still valid" >&2
fi

echo ""
echo "============================================================"
echo " Final summary"
echo "============================================================"
echo "git commit        : ${GIT_COMMIT}"
echo "finished (UTC)    : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "successful runs   : ${#SUCCESSFUL_RUNS[@]}/${TOTAL_RUNS}"
for ENTRY in ${SUCCESSFUL_RUNS[@]+"${SUCCESSFUL_RUNS[@]}"}; do
    echo "  - ${ENTRY%%=*} in ${ENTRY##*=}"
done
echo "results directory : ${OUTPUT_DIR}"
echo "summary CSV       : ${OUTPUT_DIR}/master_summary.csv"
echo "statistics CSV    : ${OUTPUT_DIR}/master_summary_stats.csv"
echo "figures           : ${OUTPUT_DIR}/figures/"
echo "============================================================"
