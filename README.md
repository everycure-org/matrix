# MATRIX - Drug Repurposing Platform

[![CI pipline](https://github.com/everycure-org/matrix/actions/workflows/matrix-ci.yml/badge.svg?branch=main)](https://github.com/everycure-org/matrix/actions/workflows/matrix-ci.yml)
[![Infrastructure Deploy](https://github.com/everycure-org/matrix/actions/workflows/infra-deploy.yml/badge.svg?branch=infra&event=push)](https://github.com/everycure-org/matrix/actions/workflows/infra-deploy.yml)
[![Documentation Page Deployment](https://github.com/everycure-org/matrix/actions/workflows/docs-deploy.yml/badge.svg)](https://github.com/everycure-org/matrix/actions/workflows/docs-deploy.yml)
[![KG Dashboard Deployment](https://github.com/everycure-org/matrix/actions/workflows/kg-dashboard-deploy.yml/badge.svg)](https://github.com/everycure-org/matrix/actions/workflows/kg-dashboard-deploy.yml)

Welcome to MATRIX! This repository contains our drug repurposing platform which includes data science pipelines, infrastructure, and documentation.

MATRIX is organized as a monorepo containing infrastructure, machine learning pipelines, applications and services. Each component resides in a dedicated directory with its own README providing detailed setup and usage instructions.

Key directories:
* `/pipelines/matrix` - 🧬 [Drug Repurposing ML Pipeline](docs/src/pipeline)
* `/infra` - 🏗️ [Infrastructure as Code (IaC) configuration](docs/src/infrastructure)
* `/apps/matrix-cli` - 🛠️ Command Line Interface Tools
* `/services` - ⚙️ Supporting Services and APIs

## 🎬 Getting Started
Ready to get started? Go to our Getting Started section

## 📚 Documentation
Please visit our [Documentation Page](http://docs.dev.everycure.org) for all details regarding the infrastructure, the repurposing pipeline or evaluation metrics.

## Contributing

We welcome and encourage all external contributions! Please see [this page to get started](./docs/src/getting_started/index.md).

## 🔗 Related Projects

- [MATRIX disease list](https://github.com/everycure-org/matrix-disease-list) - Repo to manage the MATRIX disease list.
- [MATRIX drug list](https://github.com/everycure-org/matrix-drug-list) - Repo to manage the MATRIX disease list. Note: this repository is private at the moment however will be open-sourced in the future.

## 📄 License
TODO