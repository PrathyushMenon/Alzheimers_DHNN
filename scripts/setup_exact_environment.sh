#!/usr/bin/env bash
set -euo pipefail

# Run from WSL Ubuntu 22.04:
#   cd /mnt/d/ALZ
#   sudo bash scripts/setup_exact_environment.sh

if [ "${EUID:-$(id -u)}" -ne 0 ]; then
  echo "Please run with sudo: sudo bash scripts/setup_exact_environment.sh" >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y \
  git curl wget ca-certificates build-essential cmake g++ \
  python3 python3-dev python3-pip python3-venv python3-numpy \
  dcm2niix libeigen3-dev zlib1g-dev libqt5opengl5-dev libqt5svg5-dev \
  libgl1-mesa-dev libfftw3-dev libtiff5-dev

if ! command -v mrconvert >/dev/null 2>&1; then
  if [ ! -d /opt/mrtrix3 ]; then
    git clone https://github.com/MRtrix3/mrtrix3.git /opt/mrtrix3
  fi
  cd /opt/mrtrix3
  ./configure
  ./build -j"$(nproc)"
  ln -sf /opt/mrtrix3/bin/* /usr/local/bin/
fi

if ! command -v antsRegistration >/dev/null 2>&1; then
  if [ ! -d /opt/ANTs ]; then
    git clone https://github.com/ANTsX/ANTs.git /opt/ANTs
  fi
  cmake -S /opt/ANTs -B /opt/ANTs/build -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=OFF
  cmake --build /opt/ANTs/build -j"$(nproc)"
  ln -sf /opt/ANTs/build/bin/* /usr/local/bin/
fi

python3 -m venv /mnt/d/ALZ/env/alzheimer_final
source /mnt/d/ALZ/env/alzheimer_final/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install \
  monai==1.2.0 timm==0.6.13 transformers==4.30.0 tabpfn==0.1.9 shap==0.42.1 \
  nibabel==5.1.0 antspyx==0.4.2 SimpleITK==2.2.1 pandas==1.5.3 \
  scikit-learn==1.2.2 scipy matplotlib seaborn tqdm tensorboardX torchsummary ruamel.yaml
python -m pip install clinicadl==0.1.2 clinica==0.7.6

echo "Environment installed. Activate with:"
echo "source /mnt/d/ALZ/env/alzheimer_final/bin/activate"
