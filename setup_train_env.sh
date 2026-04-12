#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ZIP_URL="https://drive.google.com/file/d/.../view?usp=sharing" bash setup_train_env.sh
# or:
#   bash setup_train_env.sh "https://drive.google.com/file/d/.../view?usp=sharing"
# Optional env vars:
#   VENV_NAME=.venv
#   ZIP_FILE=dataset.zip
#   EXTRACT_DIR=<defaults to ZIP_FILE without .zip>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ZIP_URL="${1:-${ZIP_URL:-}}"
VENV_NAME="${VENV_NAME:-.venv}"
ZIP_FILE="${ZIP_FILE:-dataset.zip}"
EXTRACT_DIR="${EXTRACT_DIR:-${ZIP_FILE%.zip}}"

if [[ "$(id -u)" -eq 0 ]]; then
  SUDO_CMD=()
else
  if command -v sudo >/dev/null 2>&1; then
    SUDO_CMD=(sudo)
  else
    echo "sudo is required for non-root user but was not found."
    exit 1
  fi
fi

if [[ -z "${ZIP_URL}" ]]; then
  echo "Missing ZIP_URL."
  echo "Example: ZIP_URL='https://drive.google.com/file/d/FILE_ID/view?usp=sharing' bash setup_train_env.sh"
  exit 1
fi

echo "[1/8] Update apt and install required system packages..."
"${SUDO_CMD[@]}" apt update
"${SUDO_CMD[@]}" apt install software-properties-common curl unzip -y

echo "[2/8] Add deadsnakes PPA and install Python 3.12..."
"${SUDO_CMD[@]}" add-apt-repository ppa:deadsnakes/ppa -y
"${SUDO_CMD[@]}" apt update
"${SUDO_CMD[@]}" apt install python3.12 python3.12-venv -y

echo "[3/8] Install pip for Python 3.12..."
curl -sS https://bootstrap.pypa.io/get-pip.py | "${SUDO_CMD[@]}" python3.12

echo "[4/8] Create virtual environment: ${VENV_NAME}"
python3.12 -m venv "${VENV_NAME}"

# shellcheck disable=SC1090
source "${VENV_NAME}/bin/activate"

echo "[5/8] Install Python packages in venv..."
python -m pip install --upgrade pip
python -m pip install --upgrade gdown
python -m pip install psutil
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "[6/8] Install project requirements..."
if [[ -f "requirement.txt" ]]; then
  python -m pip install -r requirement.txt
elif [[ -f "requirements.txt" ]]; then
  python -m pip install -r requirements.txt
else
  echo "No requirement.txt or requirements.txt found. Skipping requirements install."
fi

extract_drive_file_id() {
  local input="$1"

  if [[ "$input" =~ /file/d/([^/?]+) ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi

  if [[ "$input" =~ [\?\&]id=([^\&]+) ]]; then
    echo "${BASH_REMATCH[1]}"
    return 0
  fi

  if [[ "$input" =~ ^[A-Za-z0-9_-]{20,}$ ]]; then
    echo "$input"
    return 0
  fi

  return 1
}

echo "[7/8] Download zip with gdown..."
if python -m gdown "${ZIP_URL}" -O "${ZIP_FILE}"; then
  :
else
  echo "Direct gdown URL download failed. Trying Google Drive file id fallback..."
  if file_id="$(extract_drive_file_id "${ZIP_URL}")"; then
    python -m gdown "https://drive.google.com/uc?id=${file_id}" -O "${ZIP_FILE}"
  else
    echo "Could not parse Google Drive file id from ZIP_URL: ${ZIP_URL}"
    echo "Use a link like https://drive.google.com/file/d/FILE_ID/view or provide FILE_ID directly."
    exit 1
  fi
fi

echo "[8/8] Unzip dataset..."
mkdir -p "${EXTRACT_DIR}"
unzip -o "${ZIP_FILE}" -d "${EXTRACT_DIR}"

echo "Done."
echo "Venv: ${VENV_NAME}"
echo "Downloaded file: ${ZIP_FILE}"
echo "Extracted to: ${EXTRACT_DIR}"
echo "Activate venv with: source ${VENV_NAME}/bin/activate"
