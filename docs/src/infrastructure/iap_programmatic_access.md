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

The authentication process follows these steps:

1. **OAuth2 Configuration**: Set up OAuth2 client credentials for authentication:
   - Client ID and Secret are configured
   - Redirect URI is set to a local callback server (http://localhost:4444/hook)
   - Required scopes: "openid" and "email"

2. **User Authentication**:
   - Opens a browser window for user authentication
   - User logs in with their Google account
   - Authorization code is received via callback

3. **Token Exchange**:
   - Exchange authorization code for access and ID tokens
   - Verify ID token to confirm user identity
   - Use ID token for MLFlow API requests

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