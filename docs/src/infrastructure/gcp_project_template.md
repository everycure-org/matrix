---
title: GCP Project Setup
---

!!! info

    This page describes what a default project template looks like when set up using our `core` project setup.


We expect a separate terraform project to be created for each project, owned by the respective project teams. This way, the teams can control their own infrastructure while the core can control the whole organization.

These resources are provided out of the box. Note, we're always keen to improve this, so send PRs to the `core` repo any time

- A VPC with subnets for EU and US regions
- A `gcs` bucket and a `tf_bootstrap.json` file at the root which contains helpful data to bootstrap a project-specific terraform configuration.
- various APIs enabled ahead of time