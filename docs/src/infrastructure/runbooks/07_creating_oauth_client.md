---
title: Creating an OAuth Client
---

## Overview

This runbook describes the process of creating an OAuth client for the Matrix project. This client is used to give users access to MLFlow when using `kedro experiment`.

OAuth client creation must be done manually through the Google Cloud Console. This limitation exists because:

1. The Every Cure Matrix project's OAuth consent screen (Brand) is configured for external users
2. Google restricts programmatic OAuth client creation via API when the consent screen is set to external access


!!! info 
    Note: When an API-created internal brand is set to public, the identityAwareProxyClients.create() API will stop working, as it requires the brand to be set to internal. Therefore, you cannot create new OAuth clients via the API after an internal brand is made public.
    [Reference](https://cloud.google.com/iap/docs/programmatic-oauth-clients#:~:text=Note:%20When%20an%20API%2Dcreated,then%20click%20Submit%20for%20verification)



Google does not allow you to progrmamtically create a new OAuth client via the API after the brand is set to public. Therefore, we need to create the OAuth client manually.

## Steps

### Create the OAuth client
1. Go to the [Credentials](https://console.cloud.google.com/apis/credentials) page in the Google Cloud Console
2. Click on the `Create Credentials` button
3. Select `OAuth client ID`
4. Select `Desktop app`
5. Select a reasonable name, e.g. `matrix-cli`
6. Click on `Create`

### Create the OAuth client secret

1. Go to the [Credentials](https://console.cloud.google.com/apis/credentials) and choose your client (e.g. `matrix-cli`)
2. On the right hand panel, under "Client secrets", copy the secret
3. Store this in the GCP Secret Manager using git crypt


