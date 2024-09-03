```markdown
# PubMedBERT Embedding API

## Overview
The PubMedBERT Embedding API is a FastAPI service that generates embeddings for text inputs using the PubMedBERT model. This service is designed to be containerized with Docker and can be deployed on Kubernetes with autoscaling capabilities.

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

### Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
```

### Build the Docker Image
Use the provided `Makefile` to build the Docker image:
```bash
make build
```

### Run the API Locally
You can run the API locally using Docker:
```bash
docker run -p 8000:8000 <your-docker-image>
```

### Test the API
You can test the API using the provided `Makefile`:
```bash
make test_call
```

## API Endpoints

### POST /v1/embeddings
Generates embeddings for the provided input.

#### Request Body
```json
{
  "input": "string or array of strings",
  "model": "string (optional, defaults to microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext)"
}
```

#### Response
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [float, float, ...],
      "index": 0
    }
  ],
  "model": "string",
  "usage": {
    "prompt_tokens": integer,
    "total_tokens": integer
  }
}
```

## Deployment on Kubernetes
1. Create a Kubernetes deployment configuration for the service.
2. Set up an autoscaling group to manage the number of replicas based on load.
3. Expose the service using a LoadBalancer or NodePort for local access.

### Example Deployment Configuration
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pubmedbert-embedding-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: pubmedbert-embedding-api
  template:
    metadata:
      labels:
        app: pubmedbert-embedding-api
    spec:
      containers:
      - name: pubmedbert-embedding-api
        image: <your-docker-image>
        ports:
        - containerPort: 8000
---
apiVersion: autoscaling/v1
kind: HorizontalPodAutoscaler
metadata:
  name: pubmedbert-embedding-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pubmedbert-embedding-api
  minReplicas: 1
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80
```

## License
This project is licensed under the MIT License.

## Acknowledgments
- [FastAPI](https://fastapi.tiangolo.com/)
- [Transformers](https://huggingface.co/transformers/)
- [Docker](https://www.docker.com/)
- [Kubernetes](https://kubernetes.io/)
```
