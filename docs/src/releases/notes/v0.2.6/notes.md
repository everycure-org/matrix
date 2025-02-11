---
draft: false
date: 2025-01-15
categories:
  - Release
authors:
  - JacquesVergine
  - emil-k
  - lvijnck
  - marcello-deluca
  - jdr0887
  - matwasilewski
  - alexeistepa
  - pascalwhoop
  - piotrkan
  - oliverw1
  - Siyan-Luo
  - webyrd
  - matej-macak
---
## Breaking Changes 🛠

* **Switch to k-fold cross-validation:** The modelling pipeline now uses k-fold cross-validation for more robust model evaluation.  This significantly changes the output structure of the modelling and evaluation pipelines. (#683)
* **Refactor to remove the refit library:** The refit library has been removed, changing the way object injection and schema validation are implemented. (#811)
* **Refactor: Change the data pipeline's output format to KGX:** The data release pipeline now outputs data in KGX format, a change requiring adjustments to downstream pipelines. (#743)


## Exciting New Features 🎉

* **Implement k-fold cross-validation:** Added k-fold cross-validation to the modelling pipeline for improved model performance assessment. (#683)
* **Implement a new Spoke KG Integration:** Integrated the Spoke Knowledge Graph into the pipeline. (#772)
* **Add a parameter to abstract out the ensemble model's aggregation function:**  Allows for easier customization of ensemble model aggregation strategies. (#905)
* **Deploy Grafana and Prometheus:** Implemented Grafana and Prometheus for improved cluster and experiment monitoring and observability. (#834)
* **Add dynamic pipeline options:** Added the possibility of loading specific settings from the catalog using a resolver. (#901)
* **Add MOAs visualizer app:** This allows for more visual exploration of the results from the MOA prediction system. (#798)


## Experiments 🧪

* **Compare performance of existing models with TxGNN:** Started experiments comparing existing models' performance with TxGNN, enhancing modeling capabilities. (#586)
* **Experiment with adding timestamps to edges:** Implemented timestamps for edges (~27% complete), enabling more robust time-split validation in future. (#588)
* **Implement the ability to run a full comparison of treat scores using various embedding models:** Added the functionality to compare treat scores from different embedding methods. (#301)
* **Add additional instructions for using pyenv rather than brew to install python:**  Improved the installation guide to allow for more flexibility in installing python. (#812)


## Bugfixes 🐛

* **Fix missing edges due to missing left join during deduplication:** Resolved a bug causing missing edges after deduplication in the integration pipeline. (#781)
* **Fix missing import due to merged branch having drifted:** Resolved an import error caused by branch drift. (#823)
* **Bugfix/release make kedro hooks covered by integration tests:** Fixed an issue preventing the kedro hooks from being properly covered by the integration tests. (#900)
* **Fixes the SchemaError: `ingest_nodes` non-nullable column `name` contains null:** Addresses a schema error related to null values in the `name` column during node ingestion. (#893)
* **Fix protocol for wipe_neo script:** Corrected the Neo4j connection protocol in the wipe script. (#899)
* **Bugfix/add trigger release label to argo:** Fixed an issue with the trigger release label in Argo. (#936)


## Technical Enhancements 🧰

* **Feature/serialize release to kgx:** Added functionality to serialize data releases into the KGX format. (#743)
* **Flag to disable mlflow locally and disable by default:** Added a flag to disable MLflow tracking locally to reduce overhead during development. This is also disabled by default. (#756)
* **Improve error verbosity:** Enhanced error messages for better debugging. (#791)
* **Cleanup: Remove unused config option from `kedro run`:** Removed an unused configuration option from the `kedro run` command. (#805)
* **Refactor: use methods on sets and correct type annotations:** Improved code readability and correctness with improved type annotations. (#806)
* **Add unit tests and very minor improvement to apply_transform:** Added unit tests for improved test coverage. (#808)
* **Update virtual environment onboarding documentation:** Improved instructions on setting up the virtual environment during onboarding. (#906)
* **Refactor to remove the refit library:** Refactored the code to remove dependency on the refit library. (#811)
* **Modeling cleanup - unify splits:** Improved data management by unifying how test-train splits are generated and managed. (#907)
* **Replace argo_node function in pipeline.py with ArgoNode class:** Refactored code for improved maintainability and modularity. (#885)
* **Simplify Neo4J SSL setup:** Streamlined the process of setting up SSL/TLS for Neo4j. (#878)
* **Use pyspark.sql consistently:** Improved code consistency and clarity by using `pyspark.sql` imports. (#923)
* **Add import sorting:** Improved code readability with added import sorting. (#931)
* **Remove duplicate catalog file for data_release pipeline:** Removed a redundant catalog file, simplifying project structure. (#795)
* **Swap `kedro submit`'s counterintuitive verbose flag for a quiet flag:** Improved usability of the `kedro submit` command. (#828)
* **Swap `kedro submit`'s counterintuitive verbose flag for a quiet flag:** Improved usability of the `kedro submit` command. (#828)


## Documentation ✏️

* **Fixed created date for articles:** Corrected date formatting in release articles. (#796)
* **Math formulas added:** Implemented support for MathJax to render mathematical formulas in documentation. (#796)
* **Added google analytics:** Enabled Google Analytics tracking for website analytics. (#796)
* **Add documentation for mechanism of action codebase:** Improved documentation for the MOA codebase. (#798)
* **Add debugging documentation for VSCode:** Provided improved instructions on debugging the codebase using VS Code. (#799)
* **Update virtual environment onboarding documentation:** Updated instructions for setting up a virtual environment during onboarding. (#906)
* **Add kedro resource documentation:** Added documentation for Kedro resources in the project. (#919)
* **Onboarding fixes + Add key Jacques:** Improved onboarding documentation and added a Git-crypt key for Jacques. (#883)
* **Onboarding documentation fixes:**  Corrected some typos and improved clarity in the onboarding documentation. (#902)
* **Update common_errors.md:** Updated frequently-encountered errors and their solutions in common errors documentation. (#925)
* **Add libomp library to installation documentation:** Added libomp library to the installation instructions. (#934)
* **Add a parameter to abstract out the ensemble model's aggregation function:** Added documentation for a new parameter allowing for flexibility in how ensemble models are aggregated. (#905)
* **Add kedro resource documentation:** Added documentation about the Kedro resources in the project. (#919)
* **Onboarding documentation fixes:** Improved onboarding documentation, correcting some typos and enhancing clarity. (#902)
* **Improve installation guide:** Improved the installation guide by including more details on using `pyenv`. (#812)
* **Add documentation for Mechanism of Action (MoA) extraction pipeline:** Added detailed documentation to explain this pipeline. (#798)
* **Add debugging documentation for VS Code:** Added documentation and examples explaining how to debug with VS Code. (#799)
* **Add SILC troubleshooting document:** Added a troubleshooting document for the SILC (Service Integration, Logging, and Configuration) process. (#836)


## Newly onboarded colleagues 🚤

* Add key Jacques (#883)
* Add key Kushal (#904)
* Add key Marcello (#892)
* Add key Matej (#886)


## Other Changes

* **Remove noise for codeowners, assigning specific groups:** Improved code ownership definition by assigning specific groups instead of individuals. (#822)
* **Adds the git sha label to the workflow template and aborts submission if git state dirty:** Added a git SHA label to the workflow template and added a check for a clean git state before submission. (#771)
* **Update the `RELEASE_NAME` environment variable:** Update the `RELEASE_NAME` environment variable to a more specific version format. (#750)
* **Addressing technical debt:** The release includes fixes for missing and unused entries in the Kedro catalog and more robust handling of node category selection during integration. (#757)
* **Rename a confusing flag column name:** Renamed the `is_ground_pos` column to `in_ground_pos` for clarity. (#893)
* **Update the `RELEASE_NAME` in `.env`:** Updated instructions on setting the RELEASE_NAME env variable. (#750)
* **Improve code review practices:** Added a section on best practices for code reviews, promoting knowledge sharing and collaboration. (#757)
* **Save `not treat` and `unknown` scores for full matrix:** Added columns to store the scores for "not treat" and "unknown" probabilities to the full matrix (#853)
* **Add label hide-from-release to Release PR:** Added label to hide Release PRs from release notes. (#852)
* **Refactor: Improve code structure in `apply_transform`:** Refactored the `apply_transform` function for improved structure and clarity. (#808)
* **incrementing spoke version:** Increased the version number for the Spoke KG integration. (#914)
* **Add `--headless` flag to CLI:** Added a flag to disable user prompts. (#861)
* **Bring back old filtering logic for SemMedDB:**  Reverted changes to the SemMedDB filtering logic. (#826)
* **Onboarding fixes:** Fixed a minor issue in the onboarding documentation. (#902)
* **Fix minor typos in Kedro Extensions' documentation:** Corrected minor typos in the documentation. (#913)
* **Add protocol information to the `wipe_neo` script:** Added necessary details on the protocol used in connecting to the Neo4j database within the `wipe_neo` script. (#899)
* **Add more specific information on the error in `common_errors.md`:** Added additional troubleshooting details regarding the issues with the Spark failure, including what to look for within the logs. (#925)
* **Improve error handling in the CLI:** Updated the `run_subprocess` function to capture and display output to both stdout and stderr during streaming, improving diagnostics in cases of issues. (#827)
* **Refactor test suite:** Improved the test suite for easier maintenance and expandability. (#930)
* **Add import sorting:** Added functionality that automatically sorts the imports, improving code readability. (#931)
* **Added additional instructions for using `pyenv` rather than `brew` to install python:** Added details for alternative python installations within the onboarding documentation. (#812)
* **Swap `is_async` flag for `is_test` flag:** Changed the meaning of a flag passed into the `kedro run` command from `is_async` to `is_test`. (#811)
* **Change the `RELEASE_NAME` environment variable to a more specific version format:** Updated the `RELEASE_NAME` environment variable for more fine-grained versioning. (#750)
* **Upgrade the required java version 11 to 17 in the docs and in the docker image:** Updated the required java version to 17 for improved compatibility. (#903)
* **Add missing `knowledge_source` column to the `transform_edges` function:** Added missing column for improved data integrity. (#811)
* **Correct Argo node's output to match the single item returned by its function:** Added code to address mismatch between Argo node's output and the item returned by its function. (#844)

