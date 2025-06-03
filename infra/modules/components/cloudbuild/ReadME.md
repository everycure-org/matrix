# GCP CloudBuild

## Rationale

This document outlines the working of GCP CloudBuild in our project. We introduced CB because we saw a need to replace our Github Actions for Open Sourcing our Infra code.

---

## Key Features Introduced

### 1. **CloudBuild Integration**
- **Purpose**: Automates deployments using Google CloudBuild.
- **Implementation**:
  - Added CloudBuild modules in `main.tf` files for `dev`, `prod`, and `wg2` environments.
  - Parameters include GitHub repository details and Slack webhook URLs.
- **Resources Created**
  - Github Connection to GCP CloudBuild (connectionV2.tf)
  - IAM SA account and related permissions to deploy terraform (iam.tf)
  - create necessary secrets such as GitCrypt and Github Token needed for proper working (secret.tf)
  - Create the Trigger and Build steps for terraform apply (trigger.tf)

### 2. **Slack Notifications**
- **Purpose**: Sends deployment status updates to Slack channels.
- **Implementation**:
  - Added `slack_webhook_url` inputs in `terragrunt.hcl` files.
  - Ensures notifications are sent for deployment failures or timeouts. (Modifiable through variables).
- **Resources Created**
  - Bucket to store the config yaml and slack JSON file. (bucket.tf)
  - Cloud Run to send the notifications to slack, it listens for messages from Pub/Sub (cloud_run.tf)
  - Pub/Sub Topic that listens to cloud build status and cloud run as it's subscriber (pub_sub.tf)
  - Service Account and permissions for cloud run to access secret (sa.tf)
  - Slack Webhook URL stored in secrets (secret.tf)

---

## Notes for Future Developers

1. **CloudBuild Module**:
   - The CloudBuild module is central to deployment automation. Ensure the parameters (`project_id`, `github_repo_token`, etc.) are correctly configured for new environments.

2. **Slack Integration**:
   - Slack notifications are configured using `slack_webhook_url`. If the webhook URL changes, update the `terragrunt.hcl` files accordingly.

3. **GitHub Authentication**:
   - The workflow relies on GitHub tokens (`github_classic_token_for_cloudbuild`). Ensure these tokens are securely stored and rotated periodically. (Need future work)
---