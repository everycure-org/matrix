#!/bin/bash
set -ue

# check dependencies
command -v pigz >/dev/null 2>&1 || { echo >&2 "pigz is required but it's not installed.  Aborting."; exit 1; }
command -v gcloud >/dev/null 2>&1 || { echo >&2 "gcloud is required but it's not installed.  Aborting."; exit 1; }

# downloads data from our bucket and uploads it to the public bucket as a zip file

VERSION=${1:-v0.4.4}

echo 'deleting old data, are you sure? (y/n)'
read -r -p "Are you sure? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "Deleting old data..."
    rm -rf data/
    rm -f data.tar.gz
else
    echo "Exiting..."
    exit 1
fi

mkdir -p data
gcloud storage cp -r gs://mtrx-us-central1-hub-dev-storage/kedro/data/releases/${VERSION}/datasets/release/prm/bigquery_edges/ ./data/
gcloud storage cp -r gs://mtrx-us-central1-hub-dev-storage/kedro/data/releases/${VERSION}/datasets/release/prm/bigquery_nodes/ ./data/

echo "Compressing data..."
tar -cf - data | pigz > data.tar.gz

echo "Uploading data to public bucket..."
gcloud storage cp data.tar.gz gs://data.dev.everycure.org/versions/${VERSION}/data/data.tar.gz

echo "Done!"
