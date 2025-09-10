---
title: Create a Release
---

## Overview

This runbook outlines the steps to create a release in our GitHub repository.

## Steps to Create a Release

1. **Prepare the branch you will run the release from.**
    - Make sure the branch includes the desired data sources / parameters you want to you in your run.
    - The branch name needs to match the naming convention `release/v{semver}`, e.g. `release/v0.2.5`. Suffixes are allowed after a dash, i.e. `release/v0.2.5-alpha`
    - You git state needs to be clean, i.e. no uncommitted or untracked files. This is to make possible to have someone else running the same command, producing the same result.
2. **Determine which pipeline to run.**
    - A data release is created by running a kedro pipeline. You can run a dedicated pipeline called `data_release` or other pipeline, which contains it.
    - Consult the [pipeline registry](https://github.com/everycure-org/matrix/blob/main/pipelines/matrix/src/matrix/pipeline_registry.py) for the current pipeline definitions.
    - Currently, data release will be triggered if one of the following pipelines are run: `data_release`, `kg_release`.
2. **Trigger the pipeline.**
    - Activate the virtual environment, `source ./matrix/pipelines/matrix/.venv/bin/activate`
    - Build and run a kedro submit command, e.g.: `kedro submit --username emil --release-version v0.2.7 --pipeline kg_release`
2. **Wait for pipeline to finish.**
    - Once the pipeline finishes, a new data release PR will be created with release notes and changelog. 
2. **Review the PR that was auto-created.**
    - Review the list and check the names of the PRs to ensure they read nicely. Consider reshuffling them so they tell a good story instead of just being a list of things.
    - Manually check who has contributed and list the contributors of the month to encourage contributions through PRs (code, docs, experiment reports, etc.). See the cli command below for how to best do this
    - Upon merging the PR, the release will be publicized to the [Every Cure website](https://docs.dev.everycure.org/releases/) by another GitHub Action. It will then also be listed under the [GitHub releases](https://github.com/everycure-org/matrix/releases).
## Commands

To list contributors, use the following command:

```bash
git log v0.1..HEAD --pretty=format:"%h %ae%n%b" | \
    awk '/^[0-9a-f]+ / {hash=$1; author=$2; print hash, author} /^Co-authored-by:/ {if (match($0, /<[^>]+>/)) print hash, substr($0, RSTART+1, RLENGTH-2)}' | \
    awk '{print $2}' | \
    sort -u
```
## Best Practices

- Ensure all PRs are labeled and titled correctly before generating the release notes.
- Review and update the release notes draft to ensure clarity and completeness.
- Acknowledge all contributors to foster a collaborative environment.

## Additional Resources

- [Semantic Versioning](https://semver.org/)
- [GitHub Releases Documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
