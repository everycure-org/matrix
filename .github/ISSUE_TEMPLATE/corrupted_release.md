---
name: 🚨 Corrupted Release
about: Track and fix a corrupted KG release
title: "[CORRUPTED] Release vX.Y.Z"
labels: bug, high-priority
assignees: ''
---

## Impact Assessment
- [ ] Document affected systems/users
- [ ] Identify current mitigation (e.g., which release to use instead)
- [ ] Tag relevant stakeholders

## Investigation Checklist

### Pipeline Analysis
- [ ] Identify corrupted pipeline stage
- [ ] Review error messages in logs
- [ ] Check Argo workflow logs (find your workflow by filtering by template: `release-vX-Y-Z`)

### Code Review
- [ ] Identify last working version
- [ ] Review code changes since last working version
- [ ] Check for related issues/PRs

### Data Analysis
- [ ] Verify last valid data version
- [ ] Check input source changes
- [ ] Analyze data quality issues in outputs
- Tools to investigate data:
    - [ ] [GCS](https://console.cloud.google.com/storage/browser/mtrx-us-central1-hub-dev-storage/kedro?project=mtrx-hub-dev-3of) directories
    - [ ] [BigQuery tables](https://console.cloud.google.com/bigquery?project=everycure-dev)
    - [ ] [Neo4j](https://neo4j.dev.everycure.org/browser/) data (if applicable)

## Fix Implementation
- [ ] Backup corrupted data for investigation
- [ ] Clean up affected resources in GCS, BigQuery, and Neo4j.
- [ ] Apply and test fixes
- [ ] Update tests if needed

## Deployment
- [ ] Trigger new release
- [ ] Monitor deployment
- [ ] Verify fix resolves the issue
- [ ] Merge changes to main branch

## Post-Fix Tasks
- [ ] Document root cause
- [ ] List preventive measures
- [ ] Update documentation if needed
- [ ] Notify stakeholders of resolution

## Notes
<!-- Add any additional context, observations, or important information -->