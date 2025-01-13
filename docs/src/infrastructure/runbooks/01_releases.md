---
title: Create a Release
---

## Overview

This runbook outlines the steps to create a release in our GitHub repository.

## Steps to Create a Release

1. **Prepare the branch you will run the release from.**
    - The branch name needs to start with `release`
    - You git state needs to be clean, i.e. no uncommitted or untracked files.
2. **Determine which pipeline to run.**
    - A data release is created by running a kedro pipeline. You can run a dedicated pipeline called `data_release` or other pipeline, which contains it.
    - Consult the [pipeline registry](https://github.com/everycure-org/matrix/blob/main/pipelines/matrix/src/matrix/pipeline_registry.py) for the current pipeline definitions.
    - Currently, data release will be triggered if the following pipelines are run: `data_release`, `kg_release`.
2. **Trigger the pipeline.**
    - Activate the virtual environment, `source ./matrix/pipelines/matrix/.venv/bin/activate`
    - Build and run a kedro submit command, e.g.: `kedro submit --username emil --release-version v0.2.7 --pipeline kg_release`
2. **Wait for pipeline to finish.**
    - Once the pipeline finishes, a new data release PR will be created with release notes and changelog.

## Additional Resources

- [Semantic Versioning](https://semver.org/)
- [GitHub Releases Documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
