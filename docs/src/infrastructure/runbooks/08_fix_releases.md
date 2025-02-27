---
title: Fix a Release
---

## Overview

This runbook outlines the steps to fix a KG release.

## Steps to Fix a Release

### 1. Track the Issue
- Create a high-priority ticket in Github or Linear
- Tag relevant stakeholders
- Provide mitigation, for example which release to use instead
- Document impact: which users/systems are affected

### 2. Investigation Phase

#### Key Questions
- Which pipeline stage initiated the corruption?
- Are there relevant error messages in the logs?
- Code Analysis:
    - When was the last known working version?
    - What code changes occurred since then?
- Data Analysis:
    - When was the last valid data version?
    - Have input sources been modified?
    - Are there data quality issues in the outputs?

#### Investigation Tools
- Argo Workflow Logs: find your workflow by filtering the template using pattern: `release-vX-Y-Z-xxxxxxxx`
- [BigQuery Console](https://console.cloud.google.com/bigquery?project=everycure-dev)
- [Neo4j Browser](https://neo4j.dev.everycure.org/browser/)
- [Google Cloud Storage (GCS)](https://console.cloud.google.com/storage/browser/mtrx-us-central1-hub-dev-storage/kedro?project=mtrx-hub-dev-3of)

### 3. Apply Fixes
- Clean up corrupted data:
    - Remove affected directories from GCS
    - Drop impacted BigQuery tables
    - Backup any data needed for investigation
- Apply fixes to release branch:
    - Checkout the release branch
    - Implement and test corrections
    - Update tests if needed

### 4. Re-deploy
- Trigger a new release
- Monitor the deployment closely
- Upon successful completion:
    - Merge release branch into main
    - Document fixes applied
    - Update the incident ticket
    - Notify affected stakeholders

### 5. Post-mortem
- Document root cause
- List preventive measures
- Schedule improvements if needed