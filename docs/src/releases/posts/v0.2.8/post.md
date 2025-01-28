---
draft: false
date: 2025-01-28
categories:
  - Release
authors:
  - piotrkan
---
# Matrix Platform `v0.2.7`: Configurable Name Resolution for Improved Pipeline Flexibility

This release of the Matrix Platform enhances the preprocessing pipeline by introducing a configurable URL for the name resolution service. This change improves flexibility, testability, and maintainability.

<!-- more -->

## Key Changes:

The name resolution service, crucial for accurately identifying and mapping medical entities within the knowledge graph, is now configurable via a dedicated parameter in the `preprocessing/parameters.yml` file.

```yaml
preprocessing.name_resolution:
  url: https://name-resolution-sri.renci.org/lookup?string={name}&autocomplete=True&highlighting=False&offset=0&limit=1
```

This update replaces the previous hardcoded development URL (`https://name-resolution-sri-dev.apps.renci.org/lookup`) with a production URL (`https://name-resolution-sri.renci.org/lookup`) and establishes a parameterized configuration. This allows seamless switching between different name resolvers without modifying the core code.

The `resolve_name` function within `preprocessing/nodes.py` now accepts a `url` parameter:

```python
@retry(...)
def resolve_name(name: str, cols_to_get: List[str], url: str) -> dict:
    ...
    result = requests.get(url.format(name=name))
    ...
```

This function uses the provided URL to construct the request, enabling the use of different name resolution services based on the environment (e.g., development, testing, production).

The `create_pipeline` function in `preprocessing/pipeline.py` now passes the configured `preprocessing.name_resolution.url` parameter to the `process_medical_nodes` and `add_source_and_target_to_clinical_trails` nodes:

```python
node(
    func=nodes.process_medical_nodes,
    inputs=["preprocessing.raw.nodes", "params:preprocessing.name_resolution.url"],
    ...
),
...
node(
    func=nodes.add_source_and_target_to_clinical_trails,
    inputs={
        ...,
        "resolver_url": "params:preprocessing.name_resolution.url",
    },
    ...
)
```

This ensures that the appropriate name resolution service is utilized throughout the pipeline.

## Benefits:

* **Increased Flexibility:**  Easily switch between different name resolution services for development, testing, and production environments.
* **Improved Testability:**  Test the pipeline with different name resolvers or mock services to isolate and verify functionality.
* **Enhanced Maintainability:**  Decoupling the name resolution URL from the code simplifies updates and reduces the risk of errors.


This release improves the robustness and maintainability of the Matrix Platform's preprocessing pipeline by providing a more flexible and testable name resolution mechanism.
