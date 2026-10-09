#!/usr/bin/env bash
#
# Exercise the NoisyTVUnity-v0 path without the Unity build.
#
# The original player is a Unity executable that the upstream repository
# (https://github.com/luchris429/noisy-tv-env) distributes out of band, through
# a Google Drive folder. Downloading it is not always possible (no Drive access,
# no GPU, no display), and the binary is not redistributed here.
#
# What this script can still do is run the *python half of that path for real*:
# it installs the protocol-compatible player of the test suite
# (tests/fixtures/fake_tv_maze_player.py) as `tv_maze.x86_64` in a temporary
# directory and then runs the ordinary smoke test with
#
#     ENV_NAME = NoisyTVUnity-v0
#     NOISY_TV_UNITY_BINARY = <temporary directory>/tv_maze
#
# so the genuine `unityagents` client binds the socket, launches the player,
# speaks the JSON/frame protocol, and decodes the frames it receives. Only the
# Unity engine is simulated: the frames come from the simple corridor MDP of the
# fixture, not from the maze scene. Use it to verify the plumbing, never to
# report results about the original environment - for those, install the real
# build (scripts/install_noisy_tv_unity.sh) and run the same command with
# NOISY_TV_UNITY_BINARY pointing at it.
#
# Usage:
#   bash scripts/smoke_unity_path.sh
#   SMOKE_TOTAL_STEPS=256 bash scripts/smoke_unity_path.sh
#   RUN_AGGREGATION=0 RUN_UNIT_TESTS=0 bash scripts/smoke_unity_path.sh
#
# Configurable variables (defaults in brackets):
#   SMOKE_TOTAL_STEPS  [64]    forwarded to scripts/smoke_master.sh
#   SMOKE_NUM_ENVS     [2]     forwarded; every environment gets its own player
#   SMOKE_ROLLOUT      [8]     forwarded
#   SMOKE_BIGAN_BATCH  [8]     forwarded
#   OUTPUT_DIR         [results/smoke_unity_path]
#   VENV               [.venv when it exists, otherwise python3]  interpreter
#                      for both the smoke test and the player
#   KEEP_PLAYER        [0]     1 keeps the temporary player directory

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${ROOT_DIR}"

FIXTURE="${ROOT_DIR}/tests/fixtures/fake_tv_maze_player.py"
if [[ ! -f "${FIXTURE}" ]]; then
    echo "ERROR: ${FIXTURE} is missing; this script needs the test fixture." >&2
    exit 1
fi

if [[ -n "${VENV:-}" ]]; then
    PYTHON="${VENV}/bin/python"
elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON="${ROOT_DIR}/.venv/bin/python"
else
    PYTHON="$(command -v python3)"
fi
if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: no usable interpreter (looked for VENV and .venv/bin/python)." >&2
    exit 1
fi
echo "interpreter   : ${PYTHON}"

# Every separate player needs pillow, because the client decodes the frames.
if ! "${PYTHON}" -c "import PIL, numpy" >/dev/null 2>&1; then
    echo "ERROR: ${PYTHON} cannot import pillow/numpy, which the upstream client" >&2
    echo "       needs to decode the frames the player sends." >&2
    exit 1
fi

PLAYER_DIR="$(mktemp -d -t noisy_tv_player_XXXXXX)"
cleanup() {
    if [[ "${KEEP_PLAYER:-0}" == "1" ]]; then
        echo "player kept  : ${PLAYER_DIR}"
    else
        rm -rf "${PLAYER_DIR}"
    fi
}
trap cleanup EXIT

# The client launches `file_name + ".x86_64"` and passes only `--port`, so the
# stand-in is a launcher script that runs the fixture with this interpreter.
cat > "${PLAYER_DIR}/tv_maze.x86_64" <<LAUNCHER
#!/bin/sh
exec "${PYTHON}" "${FIXTURE}" "\$@"
LAUNCHER
chmod 755 "${PLAYER_DIR}/tv_maze.x86_64"

echo "player        : ${PLAYER_DIR}/tv_maze.x86_64 (protocol stand-in, not Unity)"
echo

NOISY_TV_UNITY_BINARY="${PLAYER_DIR}/tv_maze" \
SMOKE_ENV="${SMOKE_ENV:-NoisyTVUnity-v0}" \
SMOKE_NUM_ENVS="${SMOKE_NUM_ENVS:-2}" \
SMOKE_ROLLOUT="${SMOKE_ROLLOUT:-8}" \
SMOKE_TOTAL_STEPS="${SMOKE_TOTAL_STEPS:-64}" \
SMOKE_BIGAN_BATCH="${SMOKE_BIGAN_BATCH:-8}" \
OUTPUT_DIR="${OUTPUT_DIR:-results/smoke_unity_path}" \
PYTHON="${PYTHON}" \
    bash scripts/smoke_master.sh "$@"
