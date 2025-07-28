# How we set up the programmatic access to MLFlow through IAP

This document describes how to set up and use programmatic access to MLFlow through Google Cloud Identity-Aware Proxy (IAP).

## Infrastructure Setup

1. We created a OAuth desktop client in the Google Cloud Console. The one we created can be found [here](https://console.cloud.google.com/apis/credentials/oauthclient/938607797672-i7md7k1u3kv89e02b6d8ouo0mi9tscos.apps.googleusercontent.com?inv=1&invt=Abl9KA&project=mtrx-hub-dev-3of)
2. Then, we need to configure IAP to allow programmatic access. This is done by registering the OAuth client ID in IAP settings:

```bash
#!/bin/bash
cat << EOF > SETTING_FILE
  access_settings:
    oauth_settings:
      programmatic_clients: ["938607797672-i7md7k1u3kv89e02b6d8ouo0mi9tscos.apps.googleusercontent.com"]
EOF

gcloud iap settings set SETTING_FILE --folder=358974436967
rm SETTING_FILE
```

## Authentication Flow

### Authenticate with User Accounts Locally

The authentication process follows these steps:

1. **OAuth2 Configuration**: Set up OAuth2 client credentials for authentication:
   - Client ID and Secret are configured
   - Redirect URI is set to a local callback server (http://localhost:33333/hook)
   - Required scopes: "openid" and "email"

2. **User Authentication**:
   - Opens a browser window for user authentication
   - User logs in with their Google account
   - Authorization code is received via callback

3. **Token Exchange**:
   - Exchange authorization code for access and ID tokens
   - Verify ID token to confirm user identity
   - Use ID token for MLFlow API requests

### Authenticate with Service Accounts in GitHub Actions

When we access IAP resources in GitHub Actions, we need to set up the authentication using service accounts. The authentication process follows these steps:

1. **OAuth2 Client Configuration**:
   - Client ID and Secret are configured.

2. **Create Service Account Key**:
   - Create a new service account key for the desired service account.

3. **Add IAM Role to the Service Account**:
   - Assign the **IAP-secured Web App User** role to the service account. This role is required to access HTTPS resources that use IAP.

4. **Add the Service Account Key to Repository Secrets**:
   - Add service account key as a secret to the dev GitHub environment.

5. **Expose the Service Account Key in GitHub Actions**:
   - Expose the service account key in GitHub Actions

6. **Retrieve the Service Account IAP Token**:
   - Get the service account IAP token by reading the secret.

## Usage

The authentication flow is implemented in `matrix.utils.authentication` and can be used as follows:

```python
from matrix.utils import authentication

# Authenticate and get token
token = authentication.get_iap_token()

# Use with MLFlow
import mlflow
import os

os.environ["MLFLOW_TRACKING_TOKEN"] = token
mlflow.set_tracking_uri("https://mlflow.platform.dev.everycure.org")
```

## Security Considerations

1. OAuth credentials should be handled securely and never committed to version control
2. Tokens should be stored securely and refreshed when expired
3. Always use HTTPS for production endpoints