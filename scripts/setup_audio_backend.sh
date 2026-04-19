#!/bin/bash
# Setup audio backend for torchaudio on macOS
# This script installs libsndfile which is required for loading WAV files

set -e

echo "Setting up audio backend for torchaudio..."

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Error: Homebrew not found. Please install Homebrew from https://brew.sh"
    exit 1
fi

# Install libsndfile
echo "Installing libsndfile via Homebrew..."
brew install libsndfile

echo "✓ Audio backend setup complete!"
echo "You can now run: python scripts/prepare_data.py --root ./data/raw"
