# LiteLLM User Guide

This document provides instructions for developers on how to connect their applications to LiteLLM.

## How to Make Your Existing Application Talk to LiteLLM

LiteLLM provides an OpenAI-compatible API, allowing you to seamlessly switch your existing applications to route through it without major code changes. Simply update the base URL and API key to point to your LiteLLM instance (e.g., `https://litellm.api.prod.everycure.org` for external access or `http://litellm.litellm.svc.cluster.local:4000` for in-cluster). Use a virtual key from LiteLLM for authentication instead of direct provider keys.

### OpenAI Python Library

Replace your direct OpenAI client initialization with a custom base URL:

```python
from openai import OpenAI

# Original (direct OpenAI)
# client = OpenAI(api_key="your-openai-key")

# Updated for LiteLLM
client = OpenAI(
    api_key="your-litellm-virtual-key",  # Obtain from LiteLLM UI or API
    base_url="https://litellm.api.prod.everycure.org"  # Or in-cluster URL (litellm.litellm.svc.cluster.local:4000)
)

# Usage remains the same
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello, world!"}]
)
```

### Pydantic

Replace your direct Pydantic client initialization with a custom base URL:

```python
from pydantic_ai import Agent
from openai import AsyncOpenAI

# Original (direct provider)
# agent = Agent('openai:gpt-4o', api_key='your-openai-key')

# Updated for LiteLLM
client = AsyncOpenAI(
    api_key="your-litellm-virtual-key",
    base_url="https://litellm.api.prod.everycure.org"
)
agent = Agent('openai:gpt-4o', client=client)

# Usage remains the same
result = await agent.run('What is the capital of France?')
```

### LangChain

Replace your direct LangChain client initialization with a custom base URL:

```python
from langchain_openai import ChatOpenAI

# Original (direct OpenAI)
# llm = ChatOpenAI(api_key="your-openai-key", model="gpt-4o")

# Updated for LiteLLM
llm = ChatOpenAI(
    api_key="your-litellm-virtual-key",
    base_url="https://litellm.api.prod.everycure.org",
    model="gpt-4o"
)

# Usage remains the same
response = llm.invoke("Tell me a joke.")
```

## Sample Client Usage (Python)

> When using outside GKE, `LITELLM_BASE` should be set to `https://litellm.api.prod.everycure.org`

```python
import os, requests, json

base_url = os.getenv("LITELLM_BASE", "https://litellm.api.prod.everycure.org")
litellm_key = os.getenv("LITELLM_VIRTUAL_KEY", "")

payload = {
  "model": "gpt-4o",
  "messages": [{"role": "user", "content": "Return a JSON object with a greeting"}],
  "response_format": {"type": "json_object"}
}

resp = requests.post(
  f"{base_url}/v1/chat/completions",
  headers={
    "Authorization": f"Bearer {litellm_key}",
    "Content-Type": "application/json"
  },
  data=json.dumps(payload),
  timeout=30,
)
print(resp.status_code)
print(resp.json())
```
