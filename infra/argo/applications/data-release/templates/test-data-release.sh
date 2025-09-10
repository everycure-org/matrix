#!/usr/bin/env bash

# Allow iterating fast on the data release workflow, by submitting a
# "hello-world" Argo workflow which has the right labels to be picked up by
# Argo Events (eventsource+sensors+eventbus) and then to trigger the GitHub Actions downstream.


# It's okay if these fail, it just means they don't exist in the remote.
# Bash, without the -e option will just continue with the next command if the previous one failed.
git push  --delete origin v0.0.1-release-test
git push  --delete origin release/v0.0.1-release-test
argo submit ./infra/argo/applications/data-release/templates/TestDataReleaseWorkflow.yaml --labels git_sha=$(git rev-parse HEAD)
