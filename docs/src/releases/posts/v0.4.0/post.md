---
draft: false
date: 2025-03-06
categories:
  - Release
authors:
  - emil-k
  - eKathleenCarter
  - Siyan-Luo
  - alexeistepa
  - JacquesVergine
  - lvijnck
  - pascalwhoop
  - piotrkan
  - amyford
  - app/github-actions
  - oliverw1
  - matentzn
---

# Matrix Platform `v0.4.0`: Evidence.dev dashboard, improved experiment tracking and 

This release of Matrix includes a new [data
dashboard](https://data.dev.everycure.org/versions/latest/evidence) based on
evidence.dev, a new pipeline submission process which groups experiment runs under their
respective experiments and a scheduled daily run of the entire pipeline using a KG sample to get fast feedback on the stability of our pipeline with real data. 

<!-- more -->

## KG Dashboard

## Pipeline Submission

We updated the pipeline submission process to group experiment runs under their respective experiments. This should make it easier to track the evolution of the pipeline over time.


## Scheduled Pipeline Runs with sample data

In an effort to increase our feedback velocity, we wanted to run the pipeline more
regularly to know when integrated pieces may lead to issues. As with any complex system,
a tradeoff needs to be found between the cost of running the pipeline and the benefits of
the feedback. We decided to run the pipeline on a _sample_ of the KG to get fast feedback
on the stability of our pipeline with real data while avoiding the cost of running the
pipeline on the entirety of the KG. This should lead to ~ 90% lower costs per run while
still providing a good indicator of the pipeline's stability.

This includes two new workflows:

1. the [creation of the sample data](https://github.com/everycure-org/matrix/blob/main/.github/workflows/create-sample-release.yml) whenever a new release is created.
2. a [scheduled workflow](https://github.com/everycure-org/matrix/blob/main/.github/workflows/scheduled-sampling-pipeline.yml) that runs the pipeline on the sample data every day at 5am GMT.

For more details, check the [documentation on the sample environment](../../../onboarding/sample_environment.md).

<!-- Notes 


## What's Changed
### Exciting New Features 🎉
* Feat/run sampling pipeline on schedule by @emil-k in https://github.com/everycure-org/matrix/pull/1105
* [Infra sync] Evidence.dev infrastructure  by @pascalwhoop in https://github.com/everycure-org/matrix/pull/1112
* Quality control data for Evidence.dev by @JacquesVergine in https://github.com/everycure-org/matrix/pull/1076
* Add --nodes to Kedro submit by @lvijnck in https://github.com/everycure-org/matrix/pull/1142
* Add a summary page to Evidence with ARPA metrics by @JacquesVergine in https://github.com/everycure-org/matrix/pull/1194
### Bugfixes 🐛
* Correct the scope of information used to generate the release notes. by @Siyan-Luo in https://github.com/everycure-org/matrix/pull/1096
* Bug/add gh token for kedro submit step in GH Actions by @Siyan-Luo in https://github.com/everycure-org/matrix/pull/1132
* Modify the location of AI-generated notes file by @Siyan-Luo in https://github.com/everycure-org/matrix/pull/1129
* Debug/use git command instead of gh command by @Siyan-Luo in https://github.com/everycure-org/matrix/pull/1178
* Debug/allow bump type input from UI by @Siyan-Luo in https://github.com/everycure-org/matrix/pull/1223
### Technical Enhancements 🧰
* Remove hardcoded SILC config by @lvijnck in https://github.com/everycure-org/matrix/pull/973
* Add GitHub release dataset for drug and disease list ingestion by @JacquesVergine in https://github.com/everycure-org/matrix/pull/1050
* Create Slack notification when pipeline submission fails on GHAction by @Siyan-Luo in https://github.com/everycure-org/matrix/pull/1141
* Setup IAP OAuth for use with MLFlow by @pascalwhoop in https://github.com/everycure-org/matrix/pull/897
* Add ability to specify mlflow experiment by name by @amyford in https://github.com/everycure-org/matrix/pull/1093
* Refactor preprocessing pipeline  by @piotrkan in https://github.com/everycure-org/matrix/pull/1088
* Fix writing to the gsheets for SILC sheet by @piotrkan in https://github.com/everycure-org/matrix/pull/1193
* Allow sample run to be manually triggered by @JacquesVergine in https://github.com/everycure-org/matrix/pull/1206
* include drug and disease in release info by @emil-k in https://github.com/everycure-org/matrix/pull/1221
### Documentation ✏️
* Remove GOOGLE_CREDENTIALS env variable from installation documentation by @JacquesVergine in https://github.com/everycure-org/matrix/pull/1108
* Add documentation for disease tagging / categorisation feature by @matentzn in https://github.com/everycure-org/matrix/pull/955
* Improve sampling documentation with release specific instructions by @JacquesVergine in https://github.com/everycure-org/matrix/pull/1166
* Define process to fix a KG release by @JacquesVergine in https://github.com/everycure-org/matrix/pull/1207
* Add documentation for explaining more tags over releases by @Siyan-Luo in https://github.com/everycure-org/matrix/pull/1209
### Newly onboarded colleagues 🚤
* Create new-eKathleenCarter.asc by @eKathleenCarter in https://github.com/everycure-org/matrix/pull/1032
### Other Changes
* Fix clinical trial preprocessing nodes by @alexeistepa in https://github.com/everycure-org/matrix/pull/1039
* Fix normalizer always returning `normalization_success=True` by @piotrkan in https://github.com/everycure-org/matrix/pull/1060
* Feat/log datasets used to mlflow by @emil-k in https://github.com/everycure-org/matrix/pull/1048
* Fix mlflow metric tracking by @piotrkan in https://github.com/everycure-org/matrix/pull/1075
* Fix ec medical nodes in preprocessing by @alexeistepa in https://github.com/everycure-org/matrix/pull/1052
* Fix schema check in preprocessing pipeline by @piotrkan in https://github.com/everycure-org/matrix/pull/1082
* Update onboarding docs to include container registry auth by @amyford in https://github.com/everycure-org/matrix/pull/1081
* Bump disease list and fix release list name  by @piotrkan in https://github.com/everycure-org/matrix/pull/1072
* Better cli for quickly adding users to multiple teams by @pascalwhoop in https://github.com/everycure-org/matrix/pull/1040
* Debug/Notes and articles generation by @Siyan-Luo in https://github.com/everycure-org/matrix/pull/1059
* Fix deadlocking on subprocess calls by @oliverw1 in https://github.com/everycure-org/matrix/pull/1089
* Feat/add custom argo prometheus metric on failed workflow status by @emil-k in https://github.com/everycure-org/matrix/pull/1098
* pinned torch and re-generate requirements on mac by @oliverw1 in https://github.com/everycure-org/matrix/pull/1109
* Update BigQuery table if it exists instead of creating it by @JacquesVergine in https://github.com/everycure-org/matrix/pull/1110
* Hotfix - change version for GT in ingestion catalog by @piotrkan in https://github.com/everycure-org/matrix/pull/1116
* Setup drugmech ingestion by @lvijnck in https://github.com/everycure-org/matrix/pull/1041
* Fix integration pipeline error with missing interpolation key by @JacquesVergine in https://github.com/everycure-org/matrix/pull/1123
* Disable stability metrics (temporarily) by @piotrkan in https://github.com/everycure-org/matrix/pull/1126
* Move de-duplication to integration from preprocessing by @piotrkan in https://github.com/everycure-org/matrix/pull/1118
* Evidence.dev code & deployment & CI by @pascalwhoop in https://github.com/everycure-org/matrix/pull/1085
* Add upstream data source to Neo4j edges by @JacquesVergine in https://github.com/everycure-org/matrix/pull/1131
* Correct BQ reporting table names and change tests to cover cloud catalog by @JacquesVergine in https://github.com/everycure-org/matrix/pull/1133
* add min max to reported aggregations by @alexeistepa in https://github.com/everycure-org/matrix/pull/1152
* increasing the timeout to handle api overloading by @emil-k in https://github.com/everycure-org/matrix/pull/1146
* [mini] status badges in readme by @pascalwhoop in https://github.com/everycure-org/matrix/pull/1145
* Improved sankey on evdev dashboard by @pascalwhoop in https://github.com/everycure-org/matrix/pull/1153
* Format kedro experiment docs by @amyford in https://github.com/everycure-org/matrix/pull/1159
* Update test configuration for topological embeddings to reduce integration test duration by @lvijnck in https://github.com/everycure-org/matrix/pull/1161
* Fix/embeddins resources by @emil-k in https://github.com/everycure-org/matrix/pull/1170
* Use OAuth secret from git crypt. Add docs by @amyford in https://github.com/everycure-org/matrix/pull/1168
* Hotfix - fix make fetch_secrets missing variable and twice defined by @amyford in https://github.com/everycure-org/matrix/pull/1172
* Fix modelling bug - modelling cloud catalog  by @alexeistepa in https://github.com/everycure-org/matrix/pull/1165
* Fix catalog in ingestion by @piotrkan in https://github.com/everycure-org/matrix/pull/1176
* Revert window size to 10 for Node2Vec Embeddings by @piotrkan in https://github.com/everycure-org/matrix/pull/1184
* add rank columns by @alexeistepa in https://github.com/everycure-org/matrix/pull/1186
* Test deploy evidence.dev 0.3.3 by @JacquesVergine in https://github.com/everycure-org/matrix/pull/1190
* Resource allocation changes for embeddings pipeline by @emil-k in https://github.com/everycure-org/matrix/pull/1179
* Feat/archive mlflow runs by @amyford in https://github.com/everycure-org/matrix/pull/1181
* only log mlflow dataset if it hasn't been logged before. by @emil-k in https://github.com/everycure-org/matrix/pull/1180
* Reduce resource requirements for edge and node ingestion into Neo4j. by @oliverw1 in https://github.com/everycure-org/matrix/pull/1195
* Debug/expand mlflow hook logging by @emil-k in https://github.com/everycure-org/matrix/pull/1204


**Full Changelog**: https://github.com/everycure-org/matrix/compare/v0.3.0...v0.4.0
-->