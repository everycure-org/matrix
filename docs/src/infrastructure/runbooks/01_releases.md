---
title: Create a Release
---

## Overview

This runbook outlines the steps to create a release in our GitHub repository, either manually or via GitHub Actions.

## Steps to Create a Release Manually

1. **Prepare the branch you will run the release from.**
    - Make sure the branch includes the desired data sources / parameters you want to you in your run.
    - The branch name needs to match the naming convention `release/v{semver}`, e.g. `release/v0.2.5`. Suffixes are allowed after a dash, i.e. `release/v0.2.5-alpha`
    - You git state needs to be clean, i.e. no uncommitted or untracked files. This is to make possible to have someone else running the same command, producing the same result.
    - The release version can be a patch downgrade, but not minor or major downgrade, e.g. if `v0.2.5` is the latest official release, `v0.2.3` is allowed to trigger while `v0.1.8` is forbidden.
2. **Determine which pipeline to run.**
    - A data release is created by running a kedro pipeline. You can run a dedicated pipeline called `data_release` or other pipeline, which contains it.
    - Consult the [pipeline registry](https://github.com/everycure-org/matrix/blob/main/pipelines/matrix/src/matrix/pipeline_registry.py) for the current pipeline definitions.
    - Currently, data release will be triggered if one of the following pipelines are run: `data_release`, `kg_release`.
2. **Trigger the pipeline.**
    - Activate the virtual environment, `source ./matrix/pipelines/matrix/.venv/bin/activate`
    - Build and run a kedro submit command, e.g.: `kedro submit --username emil --release-version v0.2.7 --pipeline kg_release`
2. **Wait for pipeline to finish.**
    - Once the pipeline finishes, a new data release PR will be created with release article and changelog. 
2. **Review the PR that was auto-created.**
    - Review the list and check the names of the PRs to ensure they read nicely. Consider reshuffling them so they tell a good story instead of just being a list of things.
    - Manually check who has contributed and list the contributors of the month to encourage contributions through PRs (code, docs, experiment reports, etc.). See the cli command below for how to best do this
    - Upon merging the PR, the release will be publicized to the [Every Cure website](https://docs.dev.everycure.org/releases/) by another GitHub Action. It will then also be listed under the [GitHub releases](https://github.com/everycure-org/matrix/releases).

## Automated Release Workflow  
To streamline the release process, we use GitHub Actions for periodic **patch** and **minor** version bumps. This workflow consists of three steps:  

1. **Run Kedro kg-release via GitHub Actions.**  
    - A **weekly patch bump** and a **monthly minor bump** are scheduled to automatically trigger the Kedro pipeline submission.  

2. **Wait for pipeline to finish.**
    - Once the pipeline finishes, a new data release PR will be created with release article and changelog. 
    
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
## FAQ
### 1. **Why are there more tags than releases**
    - The tag is created and pushed during the creation of the release PR from the GitHub Actions workflow. However, the [release history webpage](https://docs.dev.everycure.org/releases/release_history/) will only be updated if the PR is merged into the main branch, bringing the release information into the main branch, whose codebase is used to build the website.
### 2. **The release pipeline is finished, but I don't see the auto-generated release PR**
This issue might be caused by a failure in the GitHub Actions workflow responsible for creating the release PR.  

#### Steps to troubleshoot and re-trigger the workflow:  
1. Navigate to the **Actions** tab in the GitHub repository.  
2. Open the **"Create Pull Request to Verify AI Summary of Release Notes"** workflow.  
3. Locate the workflow run for the release you triggered (it should be the most recent one).  
4. Inside the workflow, find the step **"Print Variables"**, and note down the printed **release version** and **Git SHA (gitref)**—these will be needed later for re-triggering.  
5. Report the failure for further investigation.  
6. Once the issue is resolved, re-trigger the workflow:  
   - Return to the UI in step 2.
   - Click **"Run workflow"**.
   - Enter the **release version** and **Git SHA** noted in step 4.  
   - Click **"Run workflow"** to initiate the process again.  

### 3. **The scheduled time has passed, but the auto-submitted release pipeline is not running**  
This issue might be due to a failure in the GitHub Actions workflow responsible for submitting the release pipeline.  

#### Steps to troubleshoot and re-trigger the workflow:  
1. Navigate to the **Actions** tab in the GitHub repository.  
2. Open the **"Periodically run kedro kg-release"** workflow.  
3. Locate the most recent workflow run for the scheduled release.  
4. Report the failure for further investigation.  
5. Once the issue is resolved, re-trigger the workflow:  
   - Return to the UI in **Step 2**.  
   - Click **"Run workflow"**.  
   - Select the version bump type (**patch** or **minor**).  
   - Click **"Run workflow"** to restart the process.  

## Best Practices

- Ensure all PRs are labeled and titled correctly before generating the release notes.
- Review and update the release notes draft to ensure clarity and completeness.
- Acknowledge all contributors to foster a collaborative environment.

## Additional Resources

- [Semantic Versioning](https://semver.org/)
- [GitHub Releases Documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
