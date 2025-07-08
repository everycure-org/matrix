The current pre-processing pipeline is highly preliminary and is ONLY used to ingest experimental nodes to our GCS bucket. The pipeline is integrated with a Google sheet for rapid hypothesis testing.

The preprocessing pipeline is mapping the names assigned by our medical team to specific IDs using name-resolver service. If you want to run the full pre-processing pipeline, you can do so by running the following command or specifying tags for a specific source:

```bash
kedro run -p preprocessing 
# or the following tags for specific sources
kedro run -p preprocessing --tag ec-clinical-trials-data
kedro run -p preprocessing --tag ec-medical-kg
```

This will then read the data from Google Sheets, normalize it, and save it to GCP under specific version specified in `globals.yaml`. Note that you don't need to re-run preprocesing every time you run e2e pipeline; you only need to re-run it if there has been any changes to the experimental data provided by the medical team.