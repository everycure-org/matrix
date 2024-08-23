# Node synonymizer

> NOTE: This is a very initial approach and we will refine this in the future.

This service provides an API wrapper around the node synonymizer for use in the pipeline. The service is implemented to maintain code separation of the pipeline and the synonimzer.

## Setup

The synonimzer requires a large database for it's operation. This database is currently stored on Google Cloud Storage (GCS). Should you want to use this service in your workflow, run the command below to retrieve the data:

```bash
make data
```


