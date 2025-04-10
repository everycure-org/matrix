# TODO: investigate if possible to get rid of this in favor of fetching this from your gcloud/kubectl config
# see: https://linear.app/everycure/issue/AIP-243/fetch-gcp-project-id-from-gcloud-kubectl-config
# Available in the cluster via a ConfigMap, but must be also present locally to:
# push the image to right gcp project repository, resolve catalog for spark datasets (when run locally).
# GCP_PROJECT_IDS = {"dev": "mtrx-hub-dev-3of", "prod": "mtrx-hub-prod-sms"}
# GCP_BUCKETS = {"dev": "mtrx-us-central1-hub-dev-storage", "prod": "mtrx-us-central1-hub-prod-storage"}
