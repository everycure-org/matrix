#!/bin/bash
set -ue

# check dependencies
command -v pigz >/dev/null 2>&1 || { echo >&2 "pigz is required but it's not installed.  Aborting."; exit 1; }
command -v gcloud >/dev/null 2>&1 || { echo >&2 "gcloud is required but it's not installed.  Aborting."; exit 1; }

# downloads data from our bucket and uploads it to the public bucket as a zip file

# Validate VERSION parameter
if [ $# -eq 0 ]; then
    echo >&2 "Error: VERSION parameter is required"
    echo >&2 "Usage: $0 VERSION"
    exit 1
fi

VERSION=$1

# Only prompt for deletion if data directory or tar file exists
if [ -d "data/" ] || [ -f "data.tar.gz" ]; then
    echo 'Found existing data files that need to be deleted'
    read -r -p "Do you want to proceed with deletion? [y/N] " response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "Deleting old data..."
        [ -d "data/" ] && rm -rf data/
        [ -f "data.tar.gz" ] && rm -f data.tar.gz
    else
        echo "Exiting..."
        exit 1
    fi
fi

mkdir -p data
gcloud storage cp -r gs://mtrx-us-central1-hub-dev-storage/kedro/data/releases/${VERSION}/datasets/release/prm/bigquery_edges/ ./data/
gcloud storage cp -r gs://mtrx-us-central1-hub-dev-storage/kedro/data/releases/${VERSION}/datasets/release/prm/bigquery_nodes/ ./data/

echo "Compressing data..."
tar -cf - data | pigz > data.tar.gz

echo "Uploading data to public bucket..."
gcloud storage cp data.tar.gz gs://data.dev.everycure.org/versions/${VERSION}/data/data.tar.gz

echo "Done!"
