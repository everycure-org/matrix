## Breaking Changes 🛠

None.


## Exciting New Features 🎉

- PR #757:  Walkthrough on how to build a new modeling pipeline in the MATRIX kedro pipeline. This likely adds a new feature or significantly enhances the pipeline creation process.
- Refactored `filtered_edges_to_kgx` and `filtered_nodes_to_kgx` functions in `nodes.py` to use a more generalized approach, improving modularity and maintainability.


## Experiments 🧪

None.


## Bugfixes 🐛

None explicitly mentioned.  However, improvements to modularity might indirectly address bugs or improve stability.


## Technical Enhancements 🧰

- PR #838: Broke data-release node logic into modular components. This improves code organization, testability, and maintainability.
- Updated `create-release-pr.yml`: Added `git config --global push.autosetupremote true`, improved handling of release version and git ref, and added a branch-off step for compliance with protected-branch policies.  The use of `jq -r` ensures that the output from `jq` is raw text to prevent errors.
- Updated `.pre-commit-config.yaml` to include `docs/src/onboarding/` in the list of directories excluded from formatting checks.
- Updated `BuildDataReleaseSensorWithReleaseVersion.yaml` to include `\ ` before the closing curly braces in the JSON payload to improve readability and prevent YAML parsing issues.
- Upgraded several dependencies in `requirements.txt` to their latest versions (e.g., `aiobotocore`, `aiosignal`, `attrs`, `botocore`, etc.).  This improves security and incorporates bug fixes and performance enhancements present in newer versions.
- Improved the `join_array_columns` function to efficiently handle the joining of multiple array columns.


## Documentation ✏️

None explicitly mentioned. The onboarding directory addition to `.pre-commit-config.yaml` suggests an implicit documentation enhancement.


## Newly onboarded colleagues 🚤

None explicitly mentioned.


## Other Changes

None.
