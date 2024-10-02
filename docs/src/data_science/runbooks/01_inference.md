---
title: Run Inference Pipeline
---
<!-- NOTE: This file was partially generated using AI assistance.  -->

# Inference Pipeline Execution with Remote MLFlow - v.0.1

The inference pipeline allows us to run prediction (i.e. inference) requests made by our medical team/stakeholders. The pipeline currently supports disease- and drug-centric as well as drug-disease specific requests. It will return report sheet in [google sheets](https://docs.google.com/spreadsheets/d/1CioSCCQxUdACn1NfWU9XRyC-9j_ERc2hmZzaDd8XgcQ/edit?gid=217784827#gid=217784827) with scores, names and associated metadata. Additionally it will produce plots visualising the output scores.

In the future, we are planning to expand the inference pipeline so that it supports:

- multi-disease / multi-drug requests - e.g. subtype-specific request where one wants to predict against several subtypes of long covid.

- therapy-area specific requests -e.g. predict against cardiovascular and neurological diseases.

## 0. Prerequisite work and best practices
Share the _Repurposing Request_ form with the person requesting inference, where they can specify drug/disease ID to predict against. You can find the form [here](https://docs.google.com/forms/d/e/1FAIpQLSecz1PUR1Bghe6YzHRB5heYiT3YdnKZq5p2GN4sYfEz3LqDFA/viewform). This input sheet from this form will be used as *preprocessing.raw.infer_sheet* entry in the data catalog

Also make sure to decide which data, model and run-name you want to use for running inference and specify them within the `.env` file (more details in Section 2.). These should be consistent with each other to avoid potential errors.

## 1. Synonymize the input 
The request might involve names/IDs in a format non-compatible with our KG. We need to resolve the input sheet and you can do it by running the following:
```bash
kedro run -p preprocessing -n clean_input_sheet
```
This generates *inference.raw.normalized_inputs* entry in data catalog that can be then used by inference pipeline. 

## 2. Set Up Environment Variables
Define the `MLFLOW_ENDPOINT`, `RUN_NAME` and `WORKFLOW_ID` in your `.env` file. The `WORKFLOW_ID` value should correspond to the MLFlow run name that you intend to use for inference.

  - **Tip:** You can find the run names by navigating to MLFlow at [MLFlow Platform](https://mlflow.platform.dev.everycure.org/).

```
# Run name which distinguishes specific experiment; it will be used as a directory name
# and experiment name within GCS, MLflow and ArgoCD.
RUN_NAME=full_matrix_run

# Workflow_id is a `sub-name` which can be found within MLFlow as a `run name`. 
# Most of the time it will have the same name as RUN_NAME.
WORKFLOW_ID= matrix-8gwc6

# The local address to the MLFlow endpoint
MLFLOW_ENDPOINT=http://127.0.0.1:5002
```

## 3. Port-Forward the MLFlow Service
To access the remote MLFlow server, you need to port-forward from your Kubernetes cluster to your local machine, as mentioned in the [onboarding documentation](https://docs.dev.everycure.org/onboarding/local-setup/). To check whether you port-forwarded successfully, you can check the `http://localhost:5002` in your browser. If successful, it will display the MLFlow UI.

## 4. Execute the Inference Pipeline
Open a new terminal and run the inference pipeline using the `--from-env cloud` (as we should never run `--env cloud` locally):

```bash
kedro run -p inference --from-env cloud
```

## 5. Terminate Port Forwarding
Once the inference pipeline is complete, stop the port-forwarding process by returning to the original terminal and pressing CTRL + C.

## 6. Version the report sheet
In order to not overwrite the report sheet when running additional repurposing request, add a time-stamp to the name which links to the `Forms` sheet. In the future, this will be done automatically.

## Troubleshooting 
One of the current limitations of the inference pipeline is the API row limit. If the pipeline fails at the `ingest_disease_list node` due to API row limits, you can restart the process from the relevant node:
```
kedro run --from-env cloud --from-nodes resolve_input_sheet
```
