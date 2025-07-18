---
title: Modify data version
---

The MATRIX pipeline is designed to be flexible with different versions of data sources. This allows you to experiment with different versions of knowledge graphs, drug/disease lists, and other data sources without modifying the core pipeline logic. The versioning system works through dependency injection - the version numbers specified in the configuration files automatically propagate through the data catalog names and file paths.

TODO: Add link to comprehensive data dictionary explaining each data source and its schema

### Configuring Data Sources in `globals.yaml`

The data sources to be used are specified in the `globals.yaml` file. These reflect the actual versions of the datasets AND how they are saved in our google cloud storage bucket as versions are encoded within a filepath. For example, see the following pathway for public KG data like ROBOKOP `${globals:paths.public_kg_raw}/robokop2/${globals:data_sources.robokop.version}/nodes_c.tsv` - the version is extracted from globals.yaml automatically within our kedro pipeline.
```yaml
# conf/base/globals.yml
data_sources:
  rtx_kg2:
    version: v2.10.0_validated
  robokop:
    version: 30fd1bfc18cd5ccb
  spoke:
    version: V5.2
  embiology:
    version: 03032025
  ec_medical_team:
    version: 20241031
  ec_clinical_trials:
    version: 20230309
  drug_list:
    version: v0.1.1
  disease_list:
    version: v0.1.1
  gt:
    version: v2.10.0_validated
  drugmech:
    version: 2.0.1
  # off_label:
  #   version: v0.1
```
!!! info 
    Note that each dataset might have its own versioning convention decided by the maintainer of the dataset. These should be explained in detail in the data dictionary. 

### Configuring Data Sources to be ingested in `settings.py`

As mentioned earlier in the guide, `settings.py` is used to specify which datasets should be processed. Here you don't specify versions to be used but just datasets which will be used within the dynamic pipeline.

```python 
#
DYNAMIC_PIPELINES_MAPPING = lambda: disable_private_datasets(
    generate_dynamic_pipeline_mapping(
        {
            "cross_validation": {
                "n_cross_val_folds": 3,
            },
            "integration": [
                {"name": "rtx_kg2", "integrate_in_kg": True, "is_private": False},
                # Following is commented out to disable it
                # {"name": "robokop", "integrate_in_kg": True, "is_private": False},
                # ... other sources
                {"name": "drug_list", "integrate_in_kg": False, "has_edges": False},
                {"name": "disease_list", "integrate_in_kg": False, "has_edges": False},
            ],
```

!!! info 
    You might have noticed several data catalog entries within the pipeline e.g. `integration.int.{source}.nodes` in the integration data catalog. `Source` variable, which is defined within `integration/pipeline.py` - this variable is extracted from `settings.py`
