#!/bin/bash
set -e
echo "[+] Installing system dependencies (FFmpeg, espeak-ng, libsndfile1, fonts)..."
apt-get update -y && apt-get install -y ffmpeg espeak-ng libsndfile1 fonts-dejavu-core

echo "[+] Upgrading pip and installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[+] Setup complete! Ready to run main.py"
