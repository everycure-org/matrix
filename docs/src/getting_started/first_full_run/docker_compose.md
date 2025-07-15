---
title: Set Up Docker for Large Data Processing
---

# Optimizing Docker for Large Data Processing

Now that you have successfully set up Docker for basic pipeline execution, this guide will help you optimize your Docker environment for handling large datasets and real-world data processing scenarios. The Matrix pipeline processes substantial amounts of data, requiring careful resource management and configuration tuning.

## Memory

When working with real data in the Matrix pipeline, you'll encounter significantly higher resource demands compared to the test environment. Therefore, you will need to optimize your docker configuration - you can do that locally through Docker Desktop application or via CLI. For running the pipeline on real data, we recommend at least 32GB RAM in memory (however for larger KGs/dimensions, you should bump the memory further). 

There are [some instructions here](https://stackoverflow.com/questions/44533319/how-to-assign-more-memory-to-docker-container) on how to optimize your Docker memory. We also recommend increasing your [swap resources](https://docs.docker.com/engine/containers/resource_constraints/). 

!!! tip "Free Disk Space"
    Note that the data products produced by our pipeline are rather large and can reach 50 GB+ in total for a single Knowledge Graph - therefore you need to ensure you have enough free disk space available


## CPU Optimization

We recommend allocating at least 6 CPU cores to your Docker container, although recommended amount would be 12. This can also be done via docker desktop settings.

!!! tip "GPU Acceleration"
    Some parts of the pipeline can be significantly accelerated using GPU hardware, particularly model fine-tuning and large-scale inference tasks (e.g. running predictions on 60M+ drug-disease pairs). If you have a GPU available, make sure to enable GPU support in Docker to take advantage of this acceleration.


[Full Data Run :material-skip-next:](./full_data_run.md){ .md-button .md-button--primary }