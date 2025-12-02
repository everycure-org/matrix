# Adding a New Provider to LiteLLM

This runbook covers how to wire a new LLM provider into the ArgoCD-managed LiteLLM deployment (`infra/argo/app-of-apps/templates/litellm.yaml`, chart `v1.79.1-stable`). Follow the steps below to add credentials, expose them to the pod, and register the provider in `model_list`.

## Prerequisites

- Confirm LiteLLM supports the provider/model family you want to expose.
- Obtain the provider API key (and any required extras such as region, base URL, or project ID).
- Keep environment key names consistent across environments (`dev` and `prod` secrets files).

## 1) Add the provider credential to Secret Manager inputs

Add the provider key to each environment’s secret manifest so External Secrets can source it:

- File: `infra/secrets/<env>/matrix/<env>_k8s_secrets.yaml`
- Naming: use lowercase for the secret store key (e.g., `mistral_api_key`) and an uppercase version for the Kubernetes secret value (e.g., `MISTRAL_API_KEY`).

Example addition (per environment):

```yaml
mistral_api_key: sk-xxx-your-provider-key
```

## 2) Map the credential into the litellm-provider-keys secret

Update the ExternalSecret template so the key is projected into the `litellm-provider-keys` Kubernetes secret consumed by the chart.

- File: `infra/argo/applications/litellm-gateway/templates/external_secrets.yaml`
- Section: `litellm-provider-keys-secret`
- Action: add a `target.template.data` entry for the environment variable and a matching `data` entry that points to the Secret Manager key.

Example snippet:

```yaml
        MISTRAL_API_KEY: "{{ `{{ .mistral_key }}` }}"
...
    - secretKey: mistral_key
      remoteRef:
        key: mistral_api_key
```

> Keep the environment variable name (`MISTRAL_API_KEY`) aligned with what you reference in `proxy_config.model_list`.

## 3) Register the provider in the LiteLLM model list

Add a new block to `proxy_config.model_list` so LiteLLM knows how to route requests to the provider.

- File: `infra/argo/app-of-apps/templates/litellm.yaml`
- Section: `proxy_config.model_list`
- Action: add a provider entry that references the environment variable from step 2 and mirrors the timeout/cache settings used elsewhere.

Example entry:

```yaml
            - model_name: mistral/*
              litellm_params:
                model: mistral/*
                api_key: os.environ/MISTRAL_API_KEY
                timeout: 300
                cache_control_injection_points:
                  - location: message
                    role: user
```

Provider-specific parameters (e.g., `api_base`, region, `vertex_project`, `vertex_location`) should be added under `litellm_params` as required by LiteLLM for that backend.

## 4) Deploy and validate

1. Commit the changes and let ArgoCD sync the updated application.
2. Confirm the projected secret contains the new key: `kubectl get secret litellm-provider-keys -n litellm -o yaml`.
3. Hit the gateway with the new model to validate end-to-end:

```bash
curl -s \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  https://litellm.api.prod.everycure.org/v1/chat/completions \
  -d '{"model": "mistral/small", "messages": [{"role": "user", "content": "ping"}]}'
```

Document any provider-specific nuances (rate limits, required headers, regional constraints) in the PR description so operators know how to support the new backend.
