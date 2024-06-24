---
title: CI
---

This page describes the various CI pipelines existing for the MATRIX project. 

## Hub Infrastructure

### Secrets files

As mentioned in the [GCP Foundations](../infrastructure/gcp_foundations.md) page, we use
`git-crypt`. Because the CI doesn't have a public-private GPG key to share with us, we
exported the symmetric key and added it to github actions' secrets. 

```
git-crypt export-key /tmp/key && cat /tmp/key | base64 && rm /tmp/key
```

## WG 1

## WG 2

## WG 3