#!/usr/bin/env bash

set -euo pipefail
IFS=$'\n\t'  # prevents many common mistakes


git push  --delete origin v0.0.1-release-test
git push  --delete origin release/v0.0.1-release-test
argo submit ./infra/argo/applications/data-release/templates/TestDataReleaseWorkflow.yaml --labels git_sha=$(git rev-parse HEAD)
