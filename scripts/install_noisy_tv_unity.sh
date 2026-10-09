#!/usr/bin/env bash
#
# Install the ORIGINAL Unity Noisy-TV environment of
# `luchris429/noisy-tv-env` -- "The Noisy TV Environment from Large-Scale Study
# of Curiosity-Driven Learning (ICLR 2019)" -- so that the training pipeline can
# run it as `NoisyTVUnity-v0` instead of the pure-python reconstruction
# (`NoisyTVMaze-v0`).
#
# Three things are needed, and the upstream repository only ships the first two:
#
#   1. the `unityagents` python client, vendored in that repository
#      (it is not on PyPI, which is why this script clones the repository),
#   2. `pillow`, because the client decodes the frames the player sends,
#   3. the player executable itself, which that project distributes out of band
#      through a Google Drive folder linked from its README.
#
# This script performs (1) and (2) automatically, then tries (3) with `gdown`
# when the folder is reachable; if the download fails (Google Drive rate-limits
# and blocks automated access regularly) it prints exactly where to put the
# build by hand. Everything lands in `third_party/noisy-tv-env`, which the
# adapter searches automatically (see environments/noisy_tv_unity.py).
#
# Usage:
#   bash scripts/install_noisy_tv_unity.sh
#   VENV=/path/to/venv DOWNLOAD=0 bash scripts/install_noisy_tv_unity.sh
#
# Configurable variables (defaults in brackets):
#   VENV           []            virtualenv to install `pillow`/`gdown` into
#   PYTHON         [python3]     interpreter used when VENV is unset
#   DESTINATION    [third_party/noisy-tv-env]
#   DRIVE_FOLDER   [1fFzDrs78teoBXtuUHJT6bvUN6MSoL_Dc]  upstream build folder
#   DOWNLOAD       [1]           attempt the Google Drive download

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${ROOT_DIR}"

DESTINATION="${DESTINATION:-third_party/noisy-tv-env}"
DRIVE_FOLDER="${DRIVE_FOLDER:-1fFzDrs78teoBXtuUHJT6bvUN6MSoL_Dc}"
DOWNLOAD="${DOWNLOAD:-1}"

if [[ -n "${VENV:-}" && -x "${VENV}/bin/python" ]]; then
    PYTHON="${VENV}/bin/python"
fi
PYTHON="${PYTHON:-python3}"

echo "============================================================"
echo " Original Unity Noisy-TV environment (luchris429/noisy-tv-env)"
echo "============================================================"
echo "destination : ${DESTINATION}"
echo "python      : ${PYTHON}"
echo "------------------------------------------------------------"

# 1. The python client shipped inside the upstream repository ----------------
if [[ -d "${DESTINATION}/unityagents" ]]; then
    echo "[1/3] upstream repository already present"
elif command -v git >/dev/null 2>&1; then
    echo "[1/3] cloning https://github.com/luchris429/noisy-tv-env"
    mkdir -p "$(dirname "${DESTINATION}")"
    git clone --depth 1 https://github.com/luchris429/noisy-tv-env.git "${DESTINATION}"
else
    echo "ERROR: git is required to fetch the vendored unityagents client" >&2
    exit 1
fi

if [[ ! -d "${DESTINATION}/unityagents" ]]; then
    echo "ERROR: ${DESTINATION}/unityagents is missing; the clone looks incomplete" >&2
    exit 1
fi

# The client targets NumPy 1.x names (np.float_, np.int_). The adapter restores
# them at import time, so no pin of NumPy is required here; pillow is.
# 2. Runtime dependencies ----------------------------------------------------
echo "[2/3] installing runtime dependencies (pillow, gdown)"
"${PYTHON}" -m pip install --quiet pillow gdown

# 3. The player executable ---------------------------------------------------
PLAYER_DIR="${DESTINATION}/build"
mkdir -p "${PLAYER_DIR}"
FOUND_PLAYER="$(find "${DESTINATION}" -maxdepth 3 \( -name 'tv_maze.x86_64' -o -name 'tv_maze.exe' -o -name 'tv_maze.app' -o -name 'tv_maze*.x86_64' \) 2>/dev/null | head -n 1 || true)"

if [[ -n "${FOUND_PLAYER}" ]]; then
    echo "[3/3] player already present: ${FOUND_PLAYER}"
elif [[ "${DOWNLOAD}" == "1" ]]; then
    echo "[3/3] downloading the player from Google Drive folder ${DRIVE_FOLDER}"
    if "${PYTHON}" -m gdown --folder "https://drive.google.com/drive/folders/${DRIVE_FOLDER}" \
            -O "${PLAYER_DIR}" --quiet 2>/dev/null; then
        FOUND_PLAYER="$(find "${PLAYER_DIR}" -maxdepth 3 \( -name 'tv_maze.x86_64' -o -name 'tv_maze.exe' -o -name 'tv_maze.app' \) 2>/dev/null | head -n 1 || true)"
    fi
fi

echo "------------------------------------------------------------"
if [[ -n "${FOUND_PLAYER}" ]]; then
    chmod +x "${FOUND_PLAYER}" 2>/dev/null || true
    echo "Player executable : ${FOUND_PLAYER}"
    echo
    echo "The adapter finds it automatically; you can use either of:"
    echo "  python main.py --env NoisyTVUnity-v0 ..."
    echo "  bash scripts/run_master_experiments.sh          # ENV_NAME=auto picks Unity"
    echo
    echo "Headless machines still need a display for the player, for example:"
    echo "  xvfb-run -a python main.py --env NoisyTVUnity-v0 ..."
else
    cat >&2 <<EOF
The python client is installed, but the player executable is still missing.

The build is NOT distributed through git; that project publishes it in the
Google Drive folder linked from its README:

  https://drive.google.com/drive/folders/${DRIVE_FOLDER}

Download the archive for your platform, extract it, and place the executable
(for Linux, the file whose name ends in .x86_64) here:

  ${PLAYER_DIR}/tv_maze.x86_64

or anywhere else and point the adapter at it:

  export NOISY_TV_UNITY_BINARY=/path/to/tv_maze          # path without the suffix
  python main.py --env NoisyTVUnity-v0 ...

Until then the pipeline keeps using the pure-python reconstruction of the same
maze (NoisyTVMaze-v0), which needs neither the player nor a display.

To check that the client, the socket protocol, and the adapter all work while
you wait for the download, run the python half of this path against the
protocol stand-in of the test suite:

  bash scripts/smoke_unity_path.sh
EOF
    exit 1
fi

echo "============================================================"
echo " Done. ENV_NAME=auto (the default) now selects NoisyTVUnity-v0."
echo "============================================================"
