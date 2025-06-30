# GCP Project Auto-Detection Feature

## Overview

This feature automatically detects the GCP project ID from the currently active gcloud/kubectl configuration, reducing the need for manual environment variable configuration when running pipelines.

## What Changed

### Before
Users needed to manually set environment variables in their `.env` file:
```bash
RUNTIME_GCP_PROJECT_ID=mtrx-hub-dev-3of
RUNTIME_GCP_BUCKET=mtrx-us-central1-hub-dev-storage
MLFLOW_URL=https://mlflow.platform.dev.everycure.org/
```

### After
The system automatically detects these values from your active gcloud configuration:
- **Project ID**: Retrieved from GCP metadata server (in GKE/GCP) or `gcloud config get-value project` (local)
- **Bucket**: Auto-determined based on the project environment (dev/prod)
- **MLflow URL**: Auto-determined based on the project environment (dev/prod)

## New Functions

### `get_gcp_project_from_metadata()`
Gets GCP project ID from the metadata server (works in GKE/GCP environments).

```python
from matrix.utils.kubernetes import get_gcp_project_from_metadata

project_id = get_gcp_project_from_metadata()
# Returns: "mtrx-hub-dev-3of" (or whatever project the GKE cluster belongs to)
```

### `get_gcp_project_from_config()`
Fetches the active GCP project ID from gcloud configuration.

```python
from matrix.utils.kubernetes import get_gcp_project_from_config

project_id = get_gcp_project_from_config()
# Returns: "mtrx-hub-dev-3of" (or whatever is active)
```

### `get_runtime_gcp_project_id()`
Gets the runtime GCP project ID using cloud-native detection methods:
1. GCP metadata server (for GKE/GCP environments)
2. gcloud CLI configuration (for local development)

```python
from matrix.utils.kubernetes import get_runtime_gcp_project_id

project_id = get_runtime_gcp_project_id()
```

### `get_runtime_mlflow_url(project_id=None)`
Auto-detects the MLflow URL based on project environment:
- Production projects (`*prod*`) → `https://mlflow.platform.prod.everycure.org/`
- Development projects → `https://mlflow.platform.dev.everycure.org/`

```python
from matrix.utils.kubernetes import get_runtime_mlflow_url

mlflow_url = get_runtime_mlflow_url()
# Or specify project explicitly:
mlflow_url = get_runtime_mlflow_url("mtrx-hub-prod-sms")
```

### `get_runtime_gcp_bucket(project_id=None)`
Auto-detects the GCP bucket based on project environment:
- Production projects → `mtrx-us-central1-hub-prod-storage`
- Development projects → `mtrx-us-central1-hub-dev-storage`

```python
from matrix.utils.kubernetes import get_runtime_gcp_bucket

bucket = get_runtime_gcp_bucket()
```

### Updated `can_talk_to_kubernetes()`
Now always auto-detects the project ID from gcloud config:

```python
from matrix.utils.kubernetes import can_talk_to_kubernetes

# Auto-detect project from gcloud config (the only way now)
can_talk_to_kubernetes()

# Optional: specify region and cluster name
can_talk_to_kubernetes(region="us-central1", cluster_name="compute-cluster")
```

## Benefits

1. **Reduced Configuration**: No need to manually set environment variables in most cases
2. **Cloud-Native Detection**: Automatically works in GKE/GCP environments via metadata server
3. **Consistency**: Always uses the same project as your active gcloud/kubectl configuration
4. **Fewer Errors**: Eliminates mismatches between gcloud config and environment variables

## How It Works

The auto-detection works by using cloud-native methods to determine the current GCP project:

1. **In GKE/GCP environments**: Uses the GCP metadata server to get the project ID from the instance metadata
2. **For local development**: When you run `gcloud container clusters get-credentials compute-cluster --project PROJECT_ID --region us-central1`, it sets your active project
3. The auto-detection functions read this active project using `gcloud config get-value project`
4. Based on the project name, the system determines if you're in dev or prod environment
5. Appropriate URLs and bucket names are automatically selected

## Service Account Impersonation Support

The auto-detection respects the `SPARK_IMPERSONATION_SERVICE_ACCOUNT` environment variable, just like the existing `can_talk_to_kubernetes` function:

```bash
export SPARK_IMPERSONATION_SERVICE_ACCOUNT=service-account@project.iam.gserviceaccount.com
```

When set, all gcloud commands will include the `--impersonate-service-account` flag.

## Migration Guide

### Breaking Change: `can_talk_to_kubernetes()` function signature
The function signature has changed to always auto-detect the project ID:

**Before:**
```python
can_talk_to_kubernetes(project="mtrx-hub-dev-3of", region="us-central1", cluster_name="compute-cluster")
```

**After:**
```python
# Project is always auto-detected from gcloud config
can_talk_to_kubernetes(region="us-central1", cluster_name="compute-cluster")
```

### For Most Users
No changes needed! The system will automatically detect your project from the GCP metadata server (in GKE/GCP) or gcloud config (locally).

### If You Need to Override Auto-Detection
The system no longer falls back to environment variables for project detection. If you need to override the auto-detected values for other components, you can still set:

```bash
# In your .env file - these override the auto-detected values for downstream components
RUNTIME_GCP_BUCKET=custom-bucket
MLFLOW_URL=https://custom.mlflow.url/
```

**Note**: `RUNTIME_GCP_PROJECT_ID` is no longer used for detection - it's only set by the system for downstream components.

### If You Have Custom Bucket Names
If your setup uses non-standard bucket names, set `RUNTIME_GCP_BUCKET` in your `.env` file:

```bash
RUNTIME_GCP_BUCKET=my-custom-bucket-name
```

## Testing

Run the test suite to verify the functionality:

```bash
cd pipelines/matrix
python -m pytest tests/test_kubernetes_auto_detection.py -v
```

## Troubleshooting

### Error: "No active GCP project found"
Solution: Set your active project using:
```bash
gcloud config set project PROJECT_ID
```

### Error: "gcloud is not installed"
Solution: Install the Google Cloud SDK:
```bash
# macOS
brew install google-cloud-sdk

# Or follow official installation guide
```

### Wrong project detected
If the auto-detection picks up the wrong project, you can:
1. Change your active gcloud project: `gcloud config set project CORRECT_PROJECT`
2. **Note**: Environment variable overrides are no longer supported for project detection

## Implementation Details

The auto-detection uses cloud-native methods and integrates with the existing `can_talk_to_kubernetes` function workflow:

1. **In GKE/GCP environments**: Uses the GCP metadata server to get the project ID directly from the instance
2. **For local development**: `can_talk_to_kubernetes` calls `gcloud container clusters get-credentials` which authenticates with a specific project
3. This same project is used by `get_gcp_project_from_config()` to maintain consistency
4. All auto-detection functions respect the same service account impersonation settings

This ensures that the project used for Kubernetes authentication matches the project used for all other operations.
