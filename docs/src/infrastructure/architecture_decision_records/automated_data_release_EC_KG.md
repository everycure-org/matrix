---
title: Automated Release Workflow
---

## Overview
The workflow automates the release process of a data-release/kg-release pipeline, allowing for both manual and scheduled triggers. It ensures seamless integration with GitHub Actions and Argo workflows while managing versioning and release updates efficiently.

![Automated data release workflow](../../assets/img/auto-data-release-pipeline.svg)

## Triggering Mechanisms
There are two ways to trigger a release:

1. **Manual Trigger**: A user manually submits a data release or kg release pipeline with a specified version. You can find more information in [this runbook](https://docs.dev.everycure.org/infrastructure/runbooks/01_releases/)
2. **Auto-Trigger via GitHub Action**: The release is automatically triggered based on a schedule:
   - Weekly patch bump
   - Monthly minor bump

## Workflow Execution
Regardless of the trigger type, the process follows these steps:

1. **Submit Argo Workflow**
   - The Argo workflow is submitted.
2. **Argo Events Processing**
   - Once the workflow is finished, `Argo EventSource` creates a data-release event.
   - `Argo EventHub` processes and forwards the event.
   - `Argo EventSensor` detects the event and triggers the next step.
3. **Trigger Repository Dispatch**
   - An HTTP POST request is sent to trigger the repository dispatch.

## Creating the Release Pull Request
Once the repository dispatch is triggered:

- A **release PR** is created via GitHub Actions.
- The GitHub action executes:
  - **Tagging**: A tag referencing the commit from which the workflow was triggered.
  - **Generating release context**, including version details and an optional release article, which are added to the PR.

## PR Handling

- **For Weekly Patch Bumps**:
  - The PR is closed: it only serves as a reminder that the `kg_release` pipeline on the main branch was working fine.
  - Weekly patches do not appear in the [Release History](https://docs.dev.everycure.org/releases/release_history/) since the release context file is not merged into `main`.

- **For Monthly Minor Bumps & Manual Releases**:
  - The PR is reviewed: the AI-generated article is revised.
  - Once approved, the PR is merged.

## Creating a Release

Once the release PR is merged:
- The **official release** is created in GitHub via GitHub Actions.
- The **CI/CD pipeline rebuilds and deploys the website** to update the [release webpage](https://docs.dev.everycure.org/releases/).

## Additional Resources
- [Create a Release](https://docs.dev.everycure.org/infrastructure/runbooks/01_releases/)
