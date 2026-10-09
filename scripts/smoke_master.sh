#!/usr/bin/env bash
#
# Master's thesis smoke test -- tiny, CPU-only, and fast (seconds, not hours).
#
# It deliberately does NOT run the expensive Montezuma experiment. It verifies
# on a small Gymnasium environment that:
#
#   1. the baseline starts            (--method state)
#   2. the proposed method starts     (--method transition)
#   3. the legacy transition path still runs (--novelty transition)
#   4. the forward model actually updates (transition/updated, MSE finite)
#   5. the intrinsic reward is finite in every logged update
#   6. the PPO update completes (finite policy/value losses, N updates logged)
#   7. the output files exist (config.json, run_info.json, metrics.jsonl, run.log)
#   8. both Master's methods share an identical PPO/environment/BiGAN config
#   9. the aggregation and figure scripts run on the smoke output
#  10. the complete unit test suite passes (legacy EME tests + new tests)
#
# Usage:
#   bash scripts/smoke_master.sh
#   SMOKE_ENV=ALE/MontezumaRevenge-v5 SMOKE_TOTAL_STEPS=256 bash scripts/smoke_master.sh
#   SMOKE_ENV=NoisyTVMaze-v0 SMOKE_TOTAL_STEPS=256 bash scripts/smoke_master.sh
#
# SMOKE_ENV=NoisyTVMaze-v0 exercises the Noisy-TV maze of "Large-Scale Study
# of Curiosity-Driven Learning" (84x84 RGB frames, sparse reward, stochastic
# television) instead of CartPole's vector observations, so the convolutional
# BiGAN/PPO path is covered end to end on the environment the thesis uses.
#
# Configurable variables (defaults in brackets):
#   SMOKE_ENV          [CartPole-v1]
#   SMOKE_NUM_ENVS     [2]
#   SMOKE_ROLLOUT      [8]
#   SMOKE_MINIBATCH    [4]
#   SMOKE_TOTAL_STEPS  [64]
#   SMOKE_BIGAN_BATCH  [8]
#   OUTPUT_DIR         [results/smoke_master]
#   PYTHON             [python3]     or set VENV=/path/to/venv
#   RUN_AGGREGATION    [1]
#   RUN_UNIT_TESTS     [1]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${ROOT_DIR}"

SMOKE_ENV="${SMOKE_ENV:-CartPole-v1}"
SMOKE_NUM_ENVS="${SMOKE_NUM_ENVS:-2}"
SMOKE_ROLLOUT="${SMOKE_ROLLOUT:-8}"
SMOKE_MINIBATCH="${SMOKE_MINIBATCH:-4}"
SMOKE_TOTAL_STEPS="${SMOKE_TOTAL_STEPS:-64}"
SMOKE_BIGAN_BATCH="${SMOKE_BIGAN_BATCH:-8}"
# The Master's experiment uses the default forward-model batch size of 64. A
# tiny smoke run collects only SMOKE_NUM_ENVS * SMOKE_ROLLOUT transitions per
# update, so the forward-model minibatch is shrunk accordingly; otherwise the
# cached latent buffer would never reach 64 samples and f_phi would never train.
SMOKE_TRANSITION_BATCH="${SMOKE_TRANSITION_BATCH:-${SMOKE_BIGAN_BATCH}}"
OUTPUT_DIR="${OUTPUT_DIR:-results/smoke_master}"
RUN_AGGREGATION="${RUN_AGGREGATION:-1}"
RUN_UNIT_TESTS="${RUN_UNIT_TESTS:-1}"

if [[ -n "${VENV:-}" && -x "${VENV}/bin/python" ]]; then
    PYTHON="${VENV}/bin/python"
fi
PYTHON="${PYTHON:-python3}"

if (( SMOKE_TOTAL_STEPS % (SMOKE_NUM_ENVS * SMOKE_ROLLOUT) != 0 )); then
    echo "ERROR: SMOKE_TOTAL_STEPS must be divisible by SMOKE_NUM_ENVS * SMOKE_ROLLOUT" >&2
    exit 1
fi
if (( SMOKE_ROLLOUT % SMOKE_MINIBATCH != 0 )); then
    echo "ERROR: SMOKE_ROLLOUT must be divisible by SMOKE_MINIBATCH" >&2
    exit 1
fi

echo "============================================================"
echo " Master's thesis smoke test (tiny CPU configuration)"
echo "============================================================"
echo "repository root : ${ROOT_DIR}"
echo "git commit      : $(git -C "${ROOT_DIR}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "environment     : ${SMOKE_ENV}   (Montezuma is intentionally NOT used here)"
echo "device          : cpu"
echo "schedule        : ${SMOKE_TOTAL_STEPS} steps = $(( SMOKE_TOTAL_STEPS / (SMOKE_NUM_ENVS * SMOKE_ROLLOUT) )) updates"
echo "forward batch   : ${SMOKE_TRANSITION_BATCH} (Master's experiment default: 64)"
echo "output dir      : ${OUTPUT_DIR}"
echo "python          : ${PYTHON}"
echo "------------------------------------------------------------"

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Output checker: validates the artifacts and numerical health of one run.
# ---------------------------------------------------------------------------
check_run() {
    local run_dir="$1"
    local mode="$2"
    "${PYTHON}" - "${run_dir}" "${mode}" <<'PY'
import json
import math
import sys
from pathlib import Path

run_directory = Path(sys.argv[1])
mode = sys.argv[2]

failures = []


def require(condition, message):
    """Record one failed requirement instead of aborting the whole check.

    Input: Boolean condition and the message describing it.
    Output: No value; failures are collected for a single report.
    Mathematical meaning: None.
    """
    if not condition:
        failures.append(message)


for name in ("config.json", "run_info.json", "metrics.jsonl", "run.log"):
    require((run_directory / name).is_file(), f"missing output file: {name}")
if failures:
    print("FAIL " + mode + ": " + "; ".join(failures))
    sys.exit(1)

configuration = json.loads((run_directory / "config.json").read_text())
run_info = json.loads((run_directory / "run_info.json").read_text())
for key in ("git_commit", "git_dirty", "timestamp", "python_version", "torch_version",
            "gymnasium_version", "ale_version", "numpy_version", "device", "method",
            "environment_id", "seed", "num_parallel_envs", "rollout_steps",
            "total_environment_steps", "key_settings"):
    require(key in run_info, f"run_info.json is missing the key {key!r}")

records = [json.loads(line) for line in (run_directory / "metrics.jsonl").read_text().splitlines() if line.strip()]
updates = [record for record in records if "update" in record["metrics"]]
require(len(updates) >= 1, "no PPO update was logged")

for record in updates:
    metrics = record["metrics"]
    for tag in ("reward/extrinsic", "reward/intrinsic", "reward/total",
                "novelty/pixel", "novelty/normalized"):
        value = metrics.get(tag)
        require(value is not None and math.isfinite(float(value)), f"{tag} is missing or not finite")
    ppo = metrics.get("ppo", {})
    for tag in ("total_loss", "policy_loss", "value_loss", "entropy", "learning_rate"):
        value = ppo.get(tag)
        require(value is not None and math.isfinite(float(value)), f"ppo/{tag} is missing or not finite")
    require(metrics.get("bigan") is not None, "BiGAN diagnostics are missing")

novelty = configuration["novelty"]
if mode == "state":
    require(novelty["novelty_type"] == "state", "config.json does not record novelty_type=state")
    require(configuration["metric_eme"]["enabled"] is False, "the EME/metric bonus must stay disabled")
    for record in updates:
        require(record["metrics"].get("transition") is None, "the baseline must not train a forward model")
elif mode == "master_transition":
    require(novelty["novelty_type"] == "transition", "config.json does not record novelty_type=transition")
    require(novelty["transition_variant"] == "master_l2", "config.json does not record transition_variant=master_l2")
    require(configuration["metric_eme"]["enabled"] is False, "the EME/metric bonus must stay disabled")
    for record in updates:
        transition = record["metrics"].get("transition")
        require(isinstance(transition, dict), "the forward model never reported an update")
        if isinstance(transition, dict):
            require(transition.get("updated") is True, "transition/updated is not True")
            mse = transition.get("forward_model_mse")
            require(mse is not None and math.isfinite(float(mse)) and float(mse) >= 0.0,
                    "transition/forward_model_mse is missing, negative, or not finite")
            error = transition.get("mean_prediction_error")
            require(error is not None and math.isfinite(float(error)),
                    "transition/mean_prediction_error is missing or not finite")
        require(float(record["metrics"]["novelty/feature"]) == 0.0,
                "the Master's method must not contain a discriminator feature term")
elif mode == "legacy_transition":
    require(novelty["novelty_type"] == "transition", "config.json does not record novelty_type=transition")
    require(novelty["transition_variant"] == "legacy", "config.json does not record transition_variant=legacy")
    for record in updates:
        transition = record["metrics"].get("transition")
        require(isinstance(transition, dict), "the legacy forward model never reported an update")
        if isinstance(transition, dict):
            for tag in ("forward_prediction_loss", "feature_matching_loss"):
                value = transition.get(tag)
                require(value is not None and math.isfinite(float(value)),
                        f"legacy transition/{tag} is missing or not finite")
else:
    failures.append(f"unknown check mode {mode!r}")

if failures:
    print(f"FAIL {mode}:")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)
print(f"OK   {mode}: {len(updates)} update(s) logged, intrinsic reward finite, artifacts complete")
PY
}

# ---------------------------------------------------------------------------
# Fairness checker: both Master's methods must share one PPO configuration.
# ---------------------------------------------------------------------------
check_fairness() {
    "${PYTHON}" - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path

state_configuration = json.loads((Path(sys.argv[1]) / "config.json").read_text())
transition_configuration = json.loads((Path(sys.argv[2]) / "config.json").read_text())

failures = []
for section in ("ppo", "bigan", "environment", "seed"):
    if state_configuration[section] != transition_configuration[section]:
        failures.append(f"configuration section {section!r} differs between the two methods")
for key in ("total_environment_steps", "use_intrinsic_reward", "resettable"):
    if state_configuration["training"][key] != transition_configuration["training"][key]:
        failures.append(f"training.{key} differs between the two methods")
if state_configuration["metric_eme"]["enabled"] or transition_configuration["metric_eme"]["enabled"]:
    failures.append("the legacy EME/metric bonus must be disabled for both Master's methods")

# Method-specific novelty fields: the state baseline uses `alpha`, the legacy
# transition variant uses `transition_alpha`, and the forward-model schedule
# only exists for the transition method. Everything that both methods share --
# above all the Equation (5) normalization settings -- must be identical.
METHOD_SPECIFIC = (
    "novelty_type",
    "transition_variant",
    "alpha",
    "transition_alpha",
    "transition_batch_size",
    "transition_hidden_dim",
    "transition_learning_rate",
    "transition_update_epochs",
    "transition_max_grad_norm",
)
state_novelty = dict(state_configuration["novelty"])
transition_novelty = dict(transition_configuration["novelty"])
for field in METHOD_SPECIFIC:
    state_novelty.pop(field, None)
    transition_novelty.pop(field, None)
if state_novelty != transition_novelty:
    failures.append(
        f"shared novelty hyperparameters differ: {state_novelty} vs {transition_novelty}"
    )
expected_forward_model = {
    "transition_hidden_dim": 256,
    "transition_learning_rate": 1e-4,
    "transition_update_epochs": 1,
    "transition_max_grad_norm": 0.5,
}
for field, value in expected_forward_model.items():
    if float(transition_configuration["novelty"][field]) != float(value):
        failures.append(f"forward-model {field} is not the documented value {value}")
if state_configuration["novelty"]["novelty_type"] != "state":
    failures.append("the baseline run did not use novelty_type=state")
if transition_configuration["novelty"]["novelty_type"] != "transition":
    failures.append("the proposed run did not use novelty_type=transition")
if transition_configuration["novelty"]["transition_variant"] != "master_l2":
    failures.append("the proposed run did not use transition_variant=master_l2")

if failures:
    print("FAIL fairness:")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)
print("OK   fairness: identical PPO/BiGAN/environment/seed/budget settings; only the intrinsic signal differs")
PY
}

run_smoke() {
    local label="$1"
    local run_dir="$2"
    shift 2
    echo ""
    echo "--- ${label} -> ${run_dir}"
    mkdir -p "${run_dir}"
    "${PYTHON}" -u main.py \
        --env "${SMOKE_ENV}" \
        --num-parallel-envs "${SMOKE_NUM_ENVS}" \
        --rollout-steps "${SMOKE_ROLLOUT}" \
        --minibatch-size "${SMOKE_MINIBATCH}" \
        --total-environment-steps "${SMOKE_TOTAL_STEPS}" \
        --bigan-batch-size "${SMOKE_BIGAN_BATCH}" \
        --device cpu \
        --disable-tensorboard \
        --disable-plots \
        --output-directory "${run_dir}" \
        "$@" 2>&1 | tee "${run_dir}/training.log"
}

STATE_DIR="${OUTPUT_DIR}/state/seed_0"
TRANSITION_DIR="${OUTPUT_DIR}/transition/seed_0"
LEGACY_DIR="${OUTPUT_DIR}/legacy_transition/seed_0"

run_smoke "Experiment A: BiGAN state novelty (--method state)" "${STATE_DIR}" \
    --method state --seed 0
check_run "${STATE_DIR}" state

run_smoke "Experiment B: BiGAN transition novelty (--method transition)" "${TRANSITION_DIR}" \
    --method transition --transition-batch-size "${SMOKE_TRANSITION_BATCH}" --seed 0
check_run "${TRANSITION_DIR}" master_transition

run_smoke "Legacy compatibility: --novelty transition" "${LEGACY_DIR}" \
    --novelty transition --seed 0
check_run "${LEGACY_DIR}" legacy_transition

echo ""
check_fairness "${STATE_DIR}" "${TRANSITION_DIR}"

if [[ "${RUN_AGGREGATION}" == "1" ]]; then
    echo ""
    echo "--- aggregation and figures on the smoke output"
    "${PYTHON}" scripts/summarize_master_results.py --results-dir "${OUTPUT_DIR}" --methods state transition
    "${PYTHON}" scripts/plot_master_results.py --results-dir "${OUTPUT_DIR}" --methods state transition
    for name in master_summary.csv master_summary_stats.csv; do
        if [[ ! -f "${OUTPUT_DIR}/${name}" ]]; then
            echo "ERROR: expected ${OUTPUT_DIR}/${name} to be created" >&2
            exit 1
        fi
    done
    if [[ ! -f "${STATE_DIR}/metrics.csv" ]]; then
        echo "ERROR: expected ${STATE_DIR}/metrics.csv to be created" >&2
        exit 1
    fi
    echo "OK   aggregation: master_summary.csv, master_summary_stats.csv, per-seed metrics.csv, figures/"
fi

if [[ "${RUN_UNIT_TESTS}" == "1" ]]; then
    echo ""
    echo "--- unit test suite (legacy EME tests + Master's transition tests)"
    "${PYTHON}" -m unittest discover -s tests -p "test_*.py"
fi

echo ""
echo "============================================================"
echo " SMOKE TEST PASSED"
echo "   baseline (--method state)            : ${STATE_DIR}"
echo "   proposed (--method transition)       : ${TRANSITION_DIR}"
echo "   legacy   (--novelty transition)      : ${LEGACY_DIR}"
echo "   summary  : ${OUTPUT_DIR}/master_summary.csv"
echo "============================================================"
