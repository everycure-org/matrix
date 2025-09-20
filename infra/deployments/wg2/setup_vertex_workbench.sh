#!/bin/bash

################################################################################
# Setup script for Vertex AI Workbench development environment
#
# This script configures a fresh Vertex AI Workbench instance with all necessary
# dependencies and configurations for the Matrix project, including:
# - Java 17 installation
# - UV package manager installation
# - Matrix repository setup with proper authentication
# - Python virtual environment configuration
# - Jupyter kernel setup
# - GCS bucket mounting
#
# The script is idempotent and will only run once, tracking its completion status
# in /home/jupyter/.vertex_setup_complete
################################################################################

set -e  # Exit on any error


# If we're not jupyter user, re-run the script as jupyter
if [ "$(whoami)" != "jupyter" ]; then
    echo "Running script as root, switching to jupyter user..."
    exec sudo -u jupyter /opt/c2d/post_start.sh
    # exec replaces the current process, so this is the last line that will run as root
fi

echo "Starting setup of development environment..."
echo "We are currently in the following directory: $(pwd)"
echo "And running as user: $(whoami)"
echo "Let's go!"
echo "---------------------------------------------------------"

export HOME="/home/jupyter"
cd $HOME

# Check if setup has been completed before
MARKER_FILE="$HOME/.vertex_setup_complete"
is_success=$(cat "$MARKER_FILE" || echo "failure")
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
export PATH="$HOME/.local/bin:$PATH"

# docker already present on the default machine image
# echo "Installing Docker..."
# curl -fsSL https://get.docker.com | sudo sh
# sudo usermod -aG docker $USER

# only clone if matrix does not yet exist
if [ ! -d "matrix" ]; then
    echo "Installing our repository and its dependencies..."
    pushd ./
    export GIT_TOKEN=$(gcloud secrets versions access latest --secret=github-token --project=mtrx-wg2-modeling-dev-9yj)
    git clone https://${GIT_TOKEN}@github.com/everycure-org/matrix.git
    make install
    cd matrix/pipelines/matrix
    # reset the remote to the normal URL, users are required to provide their own credentials to access the repo
    git remote set-url origin https://github.com/everycure-org/matrix.git
    make fetch_sa_key
    # Install the virtual environment's Python as a Jupyter kernel without activating the venv
    .venv/bin/python -m ipykernel install --user --name matrix --display-name "Python (matrix)"
    # .venv/bin/jupyter kernelspec set-default matrix

    popd
fi

# Check if the mount point is already in fstab before adding it
if ! grep -q "mtrx-us-central1-wg2-modeling-dev-storage" /etc/fstab; then
    echo "Adding GCS bucket mount point to /etc/fstab..."
    mkdir -p /home/jupyter/buckets/mtrx-us-central1-wg2-modeling-dev-storage
    echo "mtrx-us-central1-wg2-modeling-dev-storage /home/jupyter/buckets/mtrx-us-central1-wg2-modeling-dev-storage gcsfuse rw,x-systemd.requires=network-online.target,_netdev,allow_other,uid=1001,gid=1001" | sudo tee -a /etc/fstab
    sudo mount -a
else
    echo "GCS bucket mount point already exists in /etc/fstab, skipping..."
fi

# note to the user that the setup is complete
echo "success" > "$MARKER_FILE"