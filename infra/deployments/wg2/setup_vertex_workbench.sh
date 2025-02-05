#!/bin/bash
# NOTE: This script was partially generated using AI assistance.

set -e  # Exit on any error

# Check if setup has been completed before
MARKER_FILE="$HOME/.vertex_setup_complete"
is_success=$(cat "$MARKER_FILE")
if [ "$is_success" == "success" ]; then
    echo "Setup has already been completed. Skipping..."
    exit 0
fi

touch "$MARKER_FILE"
echo "Setup script failed, please check the logs and install dependencies manually " > "$MARKER_FILE"
echo "failure" > "$MARKER_FILE"

echo "Starting setup of development environment..."

echo "Updating package lists..."
sudo apt-get update

echo "Installing Java 17..."
sudo apt-get install -y openjdk-17-jre

echo "Installing UV..."
curl -LsSf https://astral.sh/uv/install.sh | sh

echo "Installing Docker..."
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER

# note to the user that the setup is complete
echo "success" > "$MARKER_FILE"