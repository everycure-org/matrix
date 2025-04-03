# Identity-Aware Proxy (IAP) Architecture

## Why IAP?

At Every Cure, we don't have VPNs or intranets. But we want to protect all our applications without a lot of work. This is where the Google Identity Aware Proxy comes in:
1. Protect all our applications from unauthorized access
2. Optionally enable them for SSO
3. Integrate seamlessly with our existing Google Workspace setup
4. Secure both human and programmatic access to our services
5. Scale efficiently as we add more services and users
6. Minimize the maintenance overhead of managing multiple auth systems

IAP offers a centralized authentication layer that protects all our web applications and APIs while leveraging our existing identity infrastructure.

## Architectural Decisions

### 1. Environment Separation

We maintain distinct IAP configurations for development and production environments. This separation serves multiple purposes:
- Enables independent access control policies for each environment (e.g., more permissive in dev, stricter in prod)
- Allows for testing IAP configuration changes safely in development
- Prevents accidental cross-environment access

### 2. Service-Specific Policies

Rather than using a single IAP policy for all services, we opted for service-specific backend policies. This granular approach:
- Enables fine-grained access control per service
- Allows for service-specific OAuth settings
- Makes it easier to audit and modify access patterns

### 3. Secret Management Strategy

We chose to manage IAP client secrets through External Secrets Operator because:
- All secrets are injected via infra as code
- Centralizes secret management in Google Secret Manager
- Enables audit trails for secret access and changes

### 4. Programmatic Access Design

## Implementation Guidelines

### Deployment of IAP in the cluster

IAP is enabled via a semi-automated approach.
### Manual steps:
1. The [Google Auth Platform](https://console.cloud.google.com/auth/overview) needs to be configured manually. This includes:
- Oauth screen (Referred to as "Branding") needs to be manually configured [here](https://console.cloud.google.com/auth/branding), giving app details and contact e-mails to app admins.
- App status (Testing/Production) and user type (Internal/External) needs to be chosen [here](https://console.cloud.google.com/auth/audience)
2. The Oauth credentials (OAuth 2.0 Client ID) needs to be manually created [here](https://console.cloud.google.com/apis/credentials). After creation, its redirect URI must be added in its settings.
3. From the generated credentials:
- `client_id` (non-sensitive) is being deployed by our terraform code as a helm value in the app of apps available to all child apps.
- `client_secret` (sensitive) value is manually added to git-crypted secrets file.

### Automated steps:
1. Credential deployment. The client secret is automatically created as a GCP secret by the terraform code.
2. IAP is automatically enabled for every argocd app, that includes the `GCPBackendPolicy` kubernetes resource in its template folder.

### Manual steps (continued):
Once the above steps are applied, the IAP will be enabled for all the applications whose service was referrenced in the `GCPBackendPolicy` resource. One can check this [here](https://console.cloud.google.com/security/iap). The access will be denied to all users by default.
1. To give access to users, you must click on each backend service in the above link, open the side pane in the UI and give the following principals the `IAP-secured Web App User ` role:
- matrix-all@everycure.org
- matrix-viewers@everycure.org
- sa-github-actions-rw@<YOUR_GCP_PROJECT_ID>.iam.gserviceaccount.com

