```markdown
# PubMedBERT Embedding API

## Overview

The PubMedBERT Embedding API is a FastAPI service that generates embeddings for text
inputs using the PubMedBERT model. This service is designed to be containerized with
Docker and can be deployed on Kubernetes with autoscaling capabilities. It's main perk
is the OpenAI spec adherence. So this can be plugged in as a hot-swap for anything that
expects OpenAI compatible embedding endpoints. We may swap this out for `vllm` in the
coming months when we move to ray. There are surely better ways than building it
ourselves.

## Features
- Generate embeddings for single or multiple text inputs.
- Utilizes the `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` model.
- Supports both JSON input and output formats.

## Requirements
- Python 3.11
- Docker
- Make
- Kubernetes (for deployment)

## Installation

Use the `Makefile` for most steps


