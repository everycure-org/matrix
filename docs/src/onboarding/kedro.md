---
title: Kedro 101
---

## Pipeline framework: Kedro

!!! info
    Kedro is an open-source framework to write modular data science code. We recommend
    checking out the [Everycure Knowledge Sharing Session on
    Kedro](https://us06web.zoom.us/rec/share/qA6wfJWiJbAZEjmD0TadG0LRi2cVxMH8pCekOLO-9aEMaPd8q8Qu7gC-O7xnDSuF.NSCc-IdN-YCn7Ysu).
    
Below is an 5 minutes intro video to Kedro

<iframe width="800" height="480" src="https://www.youtube.com/embed/PdNkECqvI58?si=_luhLzYsI3F7dQ2w&amp;start=70" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

We're using [Kedro](https://kedro.org/) as our data pipelining framework. Kedro is a rather light framework, that comes with the following [key concepts](https://docs.kedro.org/en/stable/get_started/kedro_concepts.html#):

1. __Project template__: Standard directory structure to streamline project layout, i.e., configuration, data, and pipelines.
1. __Data catalog__: A lightweight abstraction for datasets, abstracting references to the file system, in a compact configuration file.
1. __Pipelines__: A `pipeline` object abstraction, that leverages `nodes` that plug into `datasets` as defined by the data catalog[^1].
1. __Environments__: [Environments](https://docs.kedro.org/en/stable/configuration/configuration_basics.html#configuration-environments) allow for codifying the execution environment of the pipeline.
1. __Visualization__: Out-of-the-box pipeline visualization based directly on the source code.

Our core pipelines' kedro project directory can be found in `pipelines/matrix` directory with an associated `README.md` with instructions.

<!--
FUTURE: Commented this out for the time being, we may want to add this back in, but need to clearly explain how we structure our catalog (pipeline centric) vs. how kedro normally does it (layers)

### Data layer convention

Data used by our pipeline is registered in the _data catalog_. To add additional structure to the catalog items, we organise our data according to the following convention:

1. __Raw__: Data as received directly from the source, no pre-processing performed.
2. __Intermediate__: Data with simple cleaning steps applied, e.g., correct typing and column names.
3. __Primary__: Golden datasets, usually obtained by merging _intermediate_ datasets.
4. __Feature__: Primary dataset enriched with features inferred from the data, e.g., enriching an `age` column given a `date-of-birth` column.
5. __Model input__: Dataset transformed for usage by a model.
6. __Models__: Materialized models, often in the form of a pickle.
7. __Model output__: Dataset containing column where model predictions are run.
8. __Reporting__: Any datasets that provide reporting, e.g., convergence plots.


!!! tip
    We name entries in our catalog according to the following format:

    `<pipeline>.<layer>.<name>`

![](../assets/img/convention.png)

-->

[Let's get your local environment set up! :material-skip-next:](./local-setup.md){ .md-button .md-button--primary }
