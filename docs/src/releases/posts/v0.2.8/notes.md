---
draft: false
date: 2025-01-28
categories:
  - Release
authors:
  - piotrkan
---
## Other Changes

*   Updated the URL for the name resolver in the preprocessing pipeline.  This change moves from a development URL (`https://name-resolution-sri-dev.apps.renci.org/lookup`) to a production URL (`https://name-resolution-sri.renci.org/lookup`). The configuration is now parameterized, making it easier to switch between different name resolvers in the future. The `resolve_name` function now accepts a `url` parameter to allow for flexible URL usage.  This impacts the `process_medical_nodes` and `add_source_and_target_to_clinical_trails` functions within the preprocessing pipeline.

