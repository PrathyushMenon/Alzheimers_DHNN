#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
pkill -f apt-get >/dev/null 2>&1 || true
pkill -f apt >/dev/null 2>&1 || true
pkill -f dpkg >/dev/null 2>&1 || true
rm -f /var/lib/dpkg/lock-frontend /var/lib/apt/lists/lock /var/cache/apt/archives/lock
cat >/etc/apt/sources.list <<'EOF'
deb http://pl.archive.ubuntu.com/ubuntu/ jammy main restricted universe multiverse
deb http://pl.archive.ubuntu.com/ubuntu/ jammy-updates main restricted universe multiverse
deb http://pl.archive.ubuntu.com/ubuntu/ jammy-backports main restricted universe multiverse
deb http://pl.archive.ubuntu.com/ubuntu/ jammy-security main restricted universe multiverse
EOF
cat /etc/apt/sources.list
apt-get clean
apt-get update -o Acquire::ForceIPv4=true -o Acquire::Retries=5 -o Acquire::Queue-Mode=host -o Acquire::http::Pipeline-Depth=0 -o Acquire::http::Timeout=30
apt-get install -y --no-install-recommends python3-venv python3-pip python3-dev python3-numpy build-essential libeigen3-dev libfftw3-dev libpng-dev libtiff5-dev zlib1g-dev libssl-dev git curl ca-certificates wget -o Acquire::ForceIPv4=true -o Acquire::Retries=5 -o Acquire::Queue-Mode=host -o Acquire::http::Pipeline-Depth=0 -o Acquire::http::Timeout=30
