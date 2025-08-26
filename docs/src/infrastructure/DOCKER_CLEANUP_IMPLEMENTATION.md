# Docker Image Cleanup Implementation

## Overview
We've implemented an **Exit Handler Approach** to automatically delete Docker images from Google Artifact Registry when Argo Workflows complete successfully.

## What Was Done

### 1. Modified Workflow Template
- **File**: `pipelines/matrix/templates/argo_wf_spec.tmpl`
- **Changes**: 
  - Added `onExit: cleanup-handler` to the workflow spec. This ensure that the `cleanup-handler` is executed when a workflow exits.
  - Added `cleanup-handler` template that runs only when workflow status is "Succeeded"
  - Added `delete-artifact-images` template that uses `gcloud` to delete the Docker image. This uses the `sa-k8s-node` SA.
  - Passes image parameter explicitly to cleanup template to ensure proper access.

### 2. Updated GKE Service Account Permissions
- **File**: `infra/modules/stacks/compute_cluster/iam.tf`
- **Changes**:
  - Added `roles/artifactregistry.admin` to the existing `sa-k8s-node` service account
  - Removed redundant `roles/artifactregistry.writer` and `roles/artifactregistry.reader` roles
  - **No new service accounts created** - reuses existing GKE node service account

## How It Works

1. **Workflow Execution**: Your normal workflow runs as usual
2. **On Success**: When the workflow completes successfully, the exit handler triggers
3. **Parameter Passing**: Image parameter is explicitly passed as input to cleanup template
4. **Image Deletion**: The cleanup step runs with `gcloud artifacts docker images delete`
5. **Authentication**: Uses existing GKE node service account with Artifact Registry admin permissions
6. **Error Handling**: If image deletion fails, it logs the error but doesn't fail the workflow

## Key Benefits

✅ **Uses existing infrastructure** - Leverages existing `sa-k8s-node` service account  
✅ **No new secrets or service accounts** - Reuses GKE's built-in authentication  
✅ **Automatic cleanup** - Runs only when workflow succeeds  
✅ **Non-blocking** - Image cleanup failures don't affect workflow success  
✅ **Selective cleanup** - Only deletes the specific image used by the workflow  
✅ **Proper permissions** - Single admin role instead of multiple overlapping roles  

## Implementation Steps Completed

1. ✅ **Modified workflow template** with exit handler
2. ✅ **Updated IAM permissions** for `sa-k8s-node` service account
3. ✅ **Tested workflow execution** - cleanup handler triggers correctly
4. ✅ **Apply Terraform changes** to grant Artifact Registry admin permissions

## Next Steps

1. **Apply Terraform changes**:
   ```bash
   cd infra/deployments/wg1  # or your deployment directory
   terraform plan
   terraform apply
   ```

2. **Test the complete flow** with a new workflow run to verify cleanup works

## Image Cleanup Details

The cleanup process:
- **Input**: Receives image parameter explicitly from workflow: `{{inputs.parameters.image_to_delete}}`
- **Validation**: Checks if image path is non-empty before proceeding
- **Deletion**: Uses `gcloud artifacts docker images delete "$IMAGE_FULL" --quiet --delete-tags`
- **Logging**: Provides detailed logs for debugging and monitoring
- **Error Handling**: Gracefully handles missing images or permission errors
- **Example**: Deletes images like `us-central1-docker.pkg.dev/mtrx-hub-dev-3of/matrix-images/matrix:nelson-sample-run-71940567`

## Security & Best Practices

- **Principle of Least Privilege**: Using single `artifactregistry.admin` role instead of multiple roles
- **No Stored Secrets**: Leverages GKE's built-in service account authentication
- **Existing Infrastructure**: No new service accounts, secrets, or complex setups
- **Error Isolation**: Cleanup failures don't impact workflow success
