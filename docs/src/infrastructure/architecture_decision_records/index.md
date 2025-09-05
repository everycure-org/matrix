# Architecture Decision Records

Here we document the architecture decisions we have made.

## Infrastructure & Access

- [Cross-Project Orchard Data Access Implementation](cross_project_orchard_data_access.md) - Implementation of secure cross-project access to orchard datasets from Matrix platform
- [Main-Only Infrastructure Deployment](main_only_infrastructure_deployment.md) - Decision to deploy infrastructure changes only from main branch
- [Secure Private Datasets](secure-private-datasets.md) - Implementation of secure handling for private datasets

## Data & Processing

- [Automated Data Release EC KG](automated_data_release_EC_KG.md) - Automation of knowledge graph data releases
- [Improve Testing Through Sampling](improve_testing_through_sampling.md) - Enhanced testing strategy using data sampling
- [New Environment Resolver](new_env_resolver.md) - Implementation of improved environment resolution

## Performance & Operations

- [CI Optimization with Self-Hosted Runners](ci_optimization_self_hosted_runners.md) - GitHub Actions self-hosted runners deployment to reduce CI time from 30 to 15 minutes
- [Switching Pipeline Runs to Spot Instances](switching_pipeline_runs_to_spot_instances.md) - Migration to cost-effective spot instances for pipeline execution
- [OSS Storage Setup](oss-storage-setup.md) - Open source storage configuration decisions

## Historical

- [History Rewrite Pre-OSS](history_rewrite_pre_oss.md) - Documentation of codebase history cleanup before open sourcing
