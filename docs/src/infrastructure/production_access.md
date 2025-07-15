# Production Access

## Impersonating a Service Account
Once added to a group, users can impersonate a service account using the following gcloud CLI command:
```bash
gcloud auth login --update-adc --impersonate-service-account=<SERVICE_ACCOUNT_EMAIL>
gcloud config set auth/impersonate_service_account <SERVICE_ACCOUNT_EMAIL>
​```

Replace <SERVICE_ACCOUNT_EMAIL> with the service account email provided by the team.

Please make sure to add this to the `.env` in pipeline/matrix

```bash
SPARK_IMPERSONATION_SERVICE_ACCOUNT=<SERVICE_ACCOUNT_EMAIL>
```

Internal Page: https://www.notion.so/everycure/Production-Access-206b57e0137380658ee8cc6ee30de3d9