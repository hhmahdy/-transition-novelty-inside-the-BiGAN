#!/usr/bin/env bash
#
# Master's thesis experiment driver -- A3C variant.
#
#   Research question: does transition-based intrinsic reward improve
#   exploration compared with the original BiGAN-based state novelty used in
#   Adventurer?
#
#   Experiment A (baseline)  --method state       BiGAN state novelty B(s)
#   Experiment B (proposed)  --method transition  BiGAN transition novelty
#                            N_T = ||f(E(s_t),a_t) - E(s_{t+1})||_2
#
# This script runs exactly the same two conditions as
# scripts/run_master_experiments.sh, but optimises them with the asynchronous
# advantage actor-critic agent (--algorithm a3c) instead of PPO. The comparison
# itself is unchanged: the two conditions differ only in the intrinsic
# exploration signal. The environment, preprocessing, BiGAN networks, Equation
# (5) reward normalisation, training budget, and seeds are identical, and no
# hyperparameter is tuned per method.
#
# A3C reuses the project's existing quantity wherever the meaning is shared
# (gamma, GAE lambdas, entropy coefficient, value-loss coefficients,
# gradient-norm clip, learning rate, Adam epsilon, intrinsic advantage
# coefficient beta, observation preprocessing, actor/critic architecture, and
# the whole exploration pipeline). Only two quantities are A3C-specific and are
# exposed below:
#
#   N_STEPS                  n-step return horizon of one actor-learner segment
#   BIGAN_UPDATE_INTERVAL    applied actor-learner updates per BiGAN /
#                            forward-model update (the A3C analogue of PPO's
#                            one-BiGAN-update-per-rollout schedule)
#
# NUM_WORKERS is the number of actor-learners (one environment each), i.e. the
# existing --num-parallel-envs flag, so the flag keeps its meaning.
#
# The legacy EME/metric bonus and the legacy transition novelty
# (--novelty transition) are NOT part of this experiment and are never launched
# by this script.
#
# Environment: the default is the Noisy-TV maze of "Large-Scale Study of
# Curiosity-Driven Learning" (luchris429/noisy-tv-env), reconstructed in pure
# python in environments/noisy_tv_maze.py: 17 rooms and 18 corridors, 84x84 RGB
# first-person frames, six discrete actions, a sparse +1 reward 2.5 units from
# the goal sphere, and the stochastic television the paper uses to expose the
# noisy-TV failure mode.
#
#   ENV_NAME=NoisyTVMaze-v0    the reconstruction (no Unity, no ROM, CPU-friendly)
#   ENV_NAME=NoisyTVUnity-v0   the original Unity build, when the executable
#                              and the upstream `unityagents` client are present
#   ENV_NAME=ALE/MontezumaRevenge-v5   the Atari benchmark used elsewhere in the thesis
#
# Usage:
#   bash scripts/run_master_a3c.sh
#   DEVICE=cpu NUM_WORKERS=8 TOTAL_STEPS=1024000 SEEDS="0 1 2" \
#       bash scripts/run_master_a3c.sh                             # quick scale
#   ENV_NAME=NoisyTVMaze-v0 DEVICE=cuda NUM_WORKERS=16 \
#       TOTAL_STEPS=12288000 SEEDS="0 1 2" bash scripts/run_master_a3c.sh
#
#   # matched PPO control with the same script (the optimisation rule is the
#   # only difference; the geometry falls back to the PPO rollout settings)
#   ALGORITHM=ppo ROLLOUT_STEPS=32 bash scripts/run_master_a3c.sh
#
# Configurable variables (defaults in brackets):
#   ALGORITHM             [a3c]      a3c|ppo (ppo gives the matched control)
#   ENV_NAME              [NoisyTVMaze-v0]  or any registered Gymnasium ID,
#                                           e.g. NoisyTVUnity-v0,
#                                           ALE/MontezumaRevenge-v5,
#                                           CartPole-v1
#   DEVICE                [auto]     auto|cpu|cuda  (auto = cuda if available)
#   NUM_WORKERS           [8]        asynchronous actor-learners (one env each)
#   N_STEPS               [20]       A3C n-step return horizon (A3C only)
#   BIGAN_UPDATE_INTERVAL [1]        A3C updates per BiGAN update (A3C only)
#   TOTAL_STEPS           [12288000]
#   ROLLOUT_STEPS         [32]       PPO rollout geometry; A3C does not use it,
#                                    but ExperimentConfig still validates it, so
#                                    it must keep dividing TOTAL_STEPS
#   MINIBATCH_SIZE        [8]        PPO geometry; must divide ROLLOUT_STEPS
#   BIGAN_BATCH_SIZE      [64]
#   SEEDS                 [0 1 2]
#   METHODS               [state transition]
#   OUTPUT_DIR            [results/master_a3c]
#   PYTHON                [python3]  or set VENV=/path/to/venv
#   PLOT_INTERVAL_UPDATES [50]       logging frequency only, identical for both
#   DISABLE_TENSORBOARD   [0]
#   DISABLE_PLOTS         [0]
#   RUN_AGGREGATION       [1]        1 = write the CSV summaries and figures
#   OVERWRITE             [0]        1 = delete an existing run directory first
#   EXTRA_ARGS            []         passed verbatim to every run (both methods)
#
# The script never retries a failed run: a failure stops everything so that
# reproducibility problems stay visible.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${ROOT_DIR}"

ALGORITHM="${ALGORITHM:-a3c}"
ENV_NAME="${ENV_NAME:-NoisyTVMaze-v0}"
DEVICE="${DEVICE:-auto}"
NUM_WORKERS="${NUM_WORKERS:-8}"
N_STEPS="${N_STEPS:-20}"
BIGAN_UPDATE_INTERVAL="${BIGAN_UPDATE_INTERVAL:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-12288000}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-32}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-8}"
BIGAN_BATCH_SIZE="${BIGAN_BATCH_SIZE:-64}"
SEEDS="${SEEDS:-0 1 2}"
METHODS="${METHODS:-state transition}"
OUTPUT_DIR="${OUTPUT_DIR:-results/master_a3c}"
PLOT_INTERVAL_UPDATES="${PLOT_INTERVAL_UPDATES:-50}"
DISABLE_TENSORBOARD="${DISABLE_TENSORBOARD:-0}"
DISABLE_PLOTS="${DISABLE_PLOTS:-0}"
RUN_AGGREGATION="${RUN_AGGREGATION:-1}"
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

case "${ALGORITHM}" in
    a3c|ppo) ;;
    *)
        echo "ERROR: ALGORITHM must be 'a3c' or 'ppo', got '${ALGORITHM}'" >&2
        exit 1
        ;;
esac

# Steps collected between two logged optimisation updates. For A3C every
# actor-learner contributes one n-step segment per update; for PPO the whole
# rollout is the unit.
if [[ "${ALGORITHM}" == "a3c" ]]; then
    SAMPLES_PER_UPDATE=$(( NUM_WORKERS * N_STEPS ))
    SEGMENT_DESCRIPTION="${NUM_WORKERS} actor-learners x ${N_STEPS}-step segments"
else
    SAMPLES_PER_UPDATE=$(( NUM_WORKERS * ROLLOUT_STEPS ))
    SEGMENT_DESCRIPTION="${NUM_WORKERS} environments x ${ROLLOUT_STEPS}-step rollouts"
fi
# ExperimentConfig validates the PPO rollout geometry for every run, whatever
# the algorithm, so it must stay consistent even though A3C does not use it.
PPO_SAMPLES_PER_UPDATE=$(( NUM_WORKERS * ROLLOUT_STEPS ))

echo "============================================================"
echo " Master's experiment with A3C: BiGAN state vs. transition novelty"
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
echo "algorithm       : ${ALGORITHM}"
echo "environment     : ${ENV_NAME}"
echo "device          : ${DEVICE}"
echo "num workers     : ${NUM_WORKERS}"
if [[ "${ALGORITHM}" == "a3c" ]]; then
    echo "n-step horizon  : ${N_STEPS}"
    echo "BiGAN interval  : ${BIGAN_UPDATE_INTERVAL} A3C update(s)"
fi
echo "total steps     : ${TOTAL_STEPS}"
echo "PPO geometry    : rollout ${ROLLOUT_STEPS}, minibatch ${MINIBATCH_SIZE} (validated, unused by A3C)"
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
print(f"threads    {torch.get_num_threads()} (one thread per actor-learner on top)")
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

# The A3C agent and trainer must import before the first (expensive) run.
if ! "${PYTHON}" -c "
import agent.a3c, trainer_a3c, workers.a3c_worker
from config import ALGORITHMS
assert 'a3c' in ALGORITHMS
" >/dev/null 2>&1; then
    echo "ERROR: the A3C integration does not import cleanly." >&2
    echo "       Check agent/a3c.py, workers/a3c_worker.py, and trainer_a3c.py." >&2
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
if (( NUM_WORKERS <= 0 )); then
    echo "ERROR: NUM_WORKERS must be positive" >&2
    exit 1
fi
if (( N_STEPS <= 0 )); then
    echo "ERROR: N_STEPS must be positive" >&2
    exit 1
fi
if (( BIGAN_UPDATE_INTERVAL <= 0 )); then
    echo "ERROR: BIGAN_UPDATE_INTERVAL must be positive" >&2
    exit 1
fi
if (( SAMPLES_PER_UPDATE <= 0 )); then
    echo "ERROR: NUM_WORKERS * N_STEPS must be positive" >&2
    exit 1
fi
if (( TOTAL_STEPS % SAMPLES_PER_UPDATE != 0 )); then
    echo "ERROR: TOTAL_STEPS (${TOTAL_STEPS}) must be divisible by ${SEGMENT_DESCRIPTION} (${SAMPLES_PER_UPDATE})" >&2
    exit 1
fi
if (( TOTAL_STEPS % PPO_SAMPLES_PER_UPDATE != 0 )); then
    echo "ERROR: TOTAL_STEPS (${TOTAL_STEPS}) must be divisible by NUM_WORKERS * ROLLOUT_STEPS (${PPO_SAMPLES_PER_UPDATE})." >&2
    echo "       ExperimentConfig validates the PPO rollout geometry for every run, so it" >&2
    echo "       must stay consistent even when A3C is the algorithm being run." >&2
    exit 1
fi
if (( ROLLOUT_STEPS % MINIBATCH_SIZE != 0 )); then
    echo "ERROR: ROLLOUT_STEPS (${ROLLOUT_STEPS}) must be divisible by MINIBATCH_SIZE (${MINIBATCH_SIZE})" >&2
    exit 1
fi
if (( BIGAN_BATCH_SIZE > SAMPLES_PER_UPDATE )); then
    echo "NOTE: BIGAN_BATCH_SIZE (${BIGAN_BATCH_SIZE}) exceeds the ${SAMPLES_PER_UPDATE} transitions" >&2
    echo "      collected per update, so the BiGAN waits for replay instead of updating" >&2
    echo "      every interval. Both conditions behave identically, but a smaller batch is usual." >&2
fi
echo "updates per run : $(( TOTAL_STEPS / SAMPLES_PER_UPDATE ))"
echo "BiGAN updates   : $(( TOTAL_STEPS / SAMPLES_PER_UPDATE / BIGAN_UPDATE_INTERVAL )) per run and seed"
echo "------------------------------------------------------------"

mkdir -p "${OUTPUT_DIR}"

COMMON_ARGS=(
    --algorithm "${ALGORITHM}"
    --env "${ENV_NAME}"
    --num-parallel-envs "${NUM_WORKERS}"
    --rollout-steps "${ROLLOUT_STEPS}"
    --minibatch-size "${MINIBATCH_SIZE}"
    --total-environment-steps "${TOTAL_STEPS}"
    --bigan-batch-size "${BIGAN_BATCH_SIZE}"
    --device "${DEVICE}"
    --plot-interval-updates "${PLOT_INTERVAL_UPDATES}"
)
if [[ "${ALGORITHM}" == "a3c" ]]; then
    COMMON_ARGS+=(
        --a3c-n-steps "${N_STEPS}"
        --a3c-bigan-update-interval "${BIGAN_UPDATE_INTERVAL}"
    )
fi
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
        echo "[${RUN_INDEX}/${TOTAL_RUNS}] method=${METHOD} seed=${SEED} algorithm=${ALGORITHM}"
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

        # Record which optimisation rule produced the numbers, so an A3C results
        # directory can never be mistaken for a PPO one.
        RECORDED_ALGORITHM="$(
            "${PYTHON}" -c "
import json, sys
from pathlib import Path
configuration = json.loads(Path('${RUN_DIR}/config.json').read_text())
print(configuration.get('algorithm', 'ppo'))
" 2>/dev/null || echo unknown
        )"
        if [[ "${RECORDED_ALGORITHM}" != "${ALGORITHM}" ]]; then
            echo "ERROR: ${RUN_DIR}/config.json records algorithm='${RECORDED_ALGORITHM}'," >&2
            echo "       expected '${ALGORITHM}'." >&2
            exit 1
        fi
        echo "[${RUN_INDEX}/${TOTAL_RUNS}] finished ${METHOD}/seed_${SEED} (${ALGORITHM}) in ${ELAPSED}s"
        SUCCESSFUL_RUNS+=("${METHOD}/seed_${SEED}=${ELAPSED}s")
    done
done

if [[ "${RUN_AGGREGATION}" == "1" ]]; then
    echo ""
    echo "============================================================"
    echo " Aggregating results"
    echo "============================================================"
    "${PYTHON}" scripts/summarize_master_results.py --results-dir "${OUTPUT_DIR}" --methods ${METHODS}
    if ! "${PYTHON}" scripts/plot_master_results.py --results-dir "${OUTPUT_DIR}" --methods ${METHODS}; then
        echo "WARNING: figure generation failed; the CSV summaries above are still valid" >&2
    fi
else
    echo ""
    echo "RUN_AGGREGATION=0: skipping the CSV summaries and figures."
    echo "  Aggregate later with:"
    echo "    ${PYTHON} scripts/summarize_master_results.py --results-dir ${OUTPUT_DIR} --methods ${METHODS}"
    echo "    ${PYTHON} scripts/plot_master_results.py --results-dir ${OUTPUT_DIR} --methods ${METHODS}"
fi

echo ""
echo "============================================================"
echo " Final summary"
echo "============================================================"
echo "algorithm         : ${ALGORITHM}"
echo "git commit        : ${GIT_COMMIT}"
echo "finished (UTC)    : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "successful runs   : ${#SUCCESSFUL_RUNS[@]}/${TOTAL_RUNS}"
for ENTRY in ${SUCCESSFUL_RUNS[@]+"${SUCCESSFUL_RUNS[@]}"}; do
    echo "  - ${ENTRY%%=*} in ${ENTRY##*=}"
done
echo "results directory : ${OUTPUT_DIR}"
if [[ "${RUN_AGGREGATION}" == "1" ]]; then
    echo "summary CSV       : ${OUTPUT_DIR}/master_summary.csv"
    echo "statistics CSV    : ${OUTPUT_DIR}/master_summary_stats.csv"
    echo "figures           : ${OUTPUT_DIR}/figures/"
fi
echo "============================================================"
