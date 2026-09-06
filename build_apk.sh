#!/usr/bin/env bash
# ==============================================================================
# 1-Click Android APK Build Script for Kivy + NumPy On-Device AI Suite
# ==============================================================================
set -e

echo "=== [1/4] Checking System Toolchain Dependencies ==="
if ! command -v git &> /dev/null || ! command -v zip &> /dev/null; then
    echo "Installing missing prerequisites (sudo apt required)..."
    sudo apt update
    sudo apt install -y git zip unzip openjdk-17-jdk python3-pip autoconf libtool \
      pkg-config zlib1g-dev libncurses5-dev libncursesw5-dev libtinfo5 cmake \
      libffi-dev libssl-dev build-essential cython3
fi

echo "=== [2/4] Verifying Buildozer & Cython ==="
pip install --upgrade pip
pip install cython==0.29.36 buildozer virtualenv

echo "=== [3/4] Running Vector Unit Tests ==="
python3 -m unittest test_models.py

echo "=== [4/4] Starting Buildozer Cross-Compilation (ARM64) ==="
# Runs python-for-android recipe compilation and generates debug APK in bin/
buildozer android debug

echo "=============================================================================="
echo "Build Successful! Your compiled Android APK is located at:"
ls -lh bin/*.apk
echo "To sideload and run on an Android device connected via USB with Debugging:"
echo "buildozer android deploy run logcat"
echo "=============================================================================="
