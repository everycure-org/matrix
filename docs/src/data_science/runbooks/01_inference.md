---
title: Run Inference Pipeline
---
<!-- NOTE: This file was partially generated using AI assistance.  -->

# Inference Pipeline Execution with Remote MLFlow

To run the inference pipeline using models stored in a remote MLFlow server, follow these steps:

## 0. 
- Share the _Repurposing Request_ form with the person requesting inference, where they can specify drug/disease ID to predict against. You can find the form [here](https://docs.google.com/forms/d/e/1FAIpQLSecz1PUR1Bghe6YzHRB5heYiT3YdnKZq5p2GN4sYfEz3LqDFA/viewform). This input sheet from this form will be used as *preprocessing.raw.infer_sheet* entry in the data catalog

## 1. Synonymize the input 
- The request might involve names/IDs in a format non-compatible with our KG. We need to resolve the input sheet and you can do it by running the following:
```bash
kedro run -p preprocessing -n clean_input_sheet
```
This generates *inference.raw.normalized_inputs* entry in data catalog that can be then used by inference pipeline. 

## 1. Set Up Environment Variables
- Define the `MLFLOW_ENDPOINT`, `RUN_NAME` and `WORKFLOW_ID` in your `.env` file. The `WORKFLOW_ID` value should correspond to the MLFlow run name that you intend to use for inference.
  - **Tip:** You can find the run names by navigating to MLFlow at [MLFlow Platform](https://mlflow.platform.dev.everycure.org/).

```
RUN_NAME=run-sept-first-node2vec # run_name of interest from gcs/mlflow
WORKFLOW_ID= run-node2vec-iter10 # note that workflow id might be different than run_name
MLFLOW_ENDPOINT=http://127.0.0.1:5002
```

## 2. Port-Forward the MLFlow Service
To access the remote MLFlow server, you need to port-forward from your Kubernetes cluster to your local machine, as mentioned in the [onboarding documentation](https://docs.dev.everycure.org/onboarding/local-setup/). To check whether you port-forwarded successfully, you can check the http://localhost:5001 in your browser. If successful, it will display the MLFlow UI.

## 3. Execute the Inference Pipeline

Open a new terminal and run the inference pipeline using the `--from-env cloud` (as we should never run `--env cloud` locally):

```bash
kedro run -p inference --from-env cloud
```

## 4. Terminate Port Forwarding

Once the inference pipeline is complete, stop the port-forwarding process by returning to the original terminal and pressing CTRL + C.

## Troubleshooting 
One of the current limitations of the inference pipeline is the API row limit. If the pipeline fails at the `ingest_disease_list node` due to API row limits, you can restart the process from the relevant node:
```
kedro run --from-env cloud --from-nodes resolve_input_sheet
```
