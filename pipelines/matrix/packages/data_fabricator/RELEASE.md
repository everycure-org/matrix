# RELEASE Notes

## 0.4.8
- Skip hydra related tests if optional dependency not installed.

## 0.4.7
- Brix post docs update.

## 0.4.6
- Synced with Vertical W and fixed docs.

## 0.4.5
- Free upper bound of `kedro` dependency from test dependencies.

## 0.4.4
- Fix edge case when num_rows not set.

## 0.4.3
- Add case study doc for model input table example.

## 0.4.2
- Remove `nb-black` from test requirements.

## 0.4.1
- Added `JoinedColumn`to facilitate bringing data columns mapped from foreign keys.

## 0.4.0
- Extend dtypes to enable integer columns with nulls and remove `integer` key from RandomNumbers.
## 0.3.5
- Add new documentation and docstrings.

## 0.3.4
- Added `hydra_instantiate_dictionary` to replace `inject_object`.
- Updated all tests and docs to reflect the change in the `YAML` syntax.

## 0.3.3 
- Added v1 docs for `more_realism.md`, `case_study_claims_data.md`, `custom_functions.md`, `case_study_sensor_data.md`, `case_study_transactions.md`.
- Extended `Faker` column class to include `localisation` and `provider_args` faker parameters.

## 0.3.2 
- Add v0 converter function to help migrate v0 Yaml files to v1.
- Docs `additional_functionality.md` included.
- Adapt `generate_random_arrays` from v0 to v1.

## 0.3.1
- Change README doc flow for better end-user readibility.

## 0.3.0
- Add python data fabricator API and tests in v1.

## 0.2.6
- Add `generate_random_arrays` to generate random arrays.

## 0.2.5
- Remove pyspark from requirements file.

## 0.2.4
- Add pytest coverage limits

## 0.2.3
- Added CHANGELOG.md for keeping track of changes to function signatures.

## 0.2.2
- Changes to enable build in brix CI.
- Added __version__.py to remove multiple existences of version in package.
- Added a new tool to check if version in brix-mapping.yml and setup.cfg are in sync.

## 0.2.1
- Added custom setup.cfg for optional dependencies.
- Added seed to myst examples to avoid md file changes on each run.
- Update tests to follow changes made in numpy 1.24.
- Add blacken-docs to format python code in md files. 

## 0.2.0
- Implement additional logic to force dtype to be `int` where `integer: True` 

## 0.1.45
- Rename .myst to .myst.md for better editor support.

## 0.1.44
- Move data fabricator doc_reqs to test_reqs. This mitigates an issue where package users where unnecessarily force to install specific other package versions (e.g. of kedro).

## 0.1.43
- Added `08_kedro_integration` use case to `data_fabricator` package.

## 0.1.42
- Introduce jupyterbook mechanism. 

## 0.1.41
- Added `seed` argument to `MockDataGenerator` class.
