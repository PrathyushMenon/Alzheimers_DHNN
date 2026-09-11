#!/usr/bin/env bash
set -euo pipefail

echo "Running WSL setup for pipeline reproducibility"
echo "This script installs system deps, MRtrix3, ANTs (build), dcm2niix, and creates a Python venv. Run as a normal user inside WSL Ubuntu-22.04."

# Update & basics
sudo apt update && sudo apt install -y build-essential git wget curl ca-certificates cmake python3-venv python3-pip \
    pkg-config libeigen3-dev zlib1g-dev libfftw3-dev libpng-dev libtiff5-dev libopenblas-dev liblapack-dev

# dcm2niix (fast install)
if ! command -v dcm2niix >/dev/null 2>&1; then
  echo "Installing dcm2niix..."
  sudo apt install -y dcm2niix || true
fi

# MRtrix3 - build from source
if ! command -v mrconvert >/dev/null 2>&1; then
  echo "Installing MRtrix3 (will clone and build in ~/mrtrix3)..."
  cd "$HOME"
  rm -rf mrtrix3
  git clone --depth 1 https://github.com/MRtrix3/mrtrix3.git mrtrix3
  cd mrtrix3
  ./configure
  ./build -j$(nproc)
  echo "Adding MRtrix3 to PATH in ~/.profile"
  grep -q 'mrtrix3/bin' ~/.profile || echo 'export PATH="$HOME/mrtrix3/bin:$PATH"' >> ~/.profile
fi

# ANTs - build via GitHub (may take long)
if ! command -v antsRegistration >/dev/null 2>&1; then
  echo "Installing ANTs (build from source)..."
  sudo apt install -y libx11-dev libxt-dev libxext-dev libpng-dev libjpeg-dev libtiff5-dev libcurl4-openssl-dev
  cd "$HOME"
  rm -rf ANTs
  git clone --depth 1 https://github.com/ANTsX/ANTs.git ANTs
  cd ANTs
  mkdir -p build && cd build
  cmake -DBUILD_SHARED_LIBS=ON ..
  make -j$(nproc)
  sudo make install || true
fi

echo "Create python venv at ./venv if missing"
if [ ! -d ./venv ]; then
  python3 -m venv ./venv
  ./venv/bin/pip install --upgrade pip
fi

echo "WSL setup complete. Activate venv with: source ./venv/bin/activate"
