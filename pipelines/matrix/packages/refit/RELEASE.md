# Release History

## 0.5.24
- Add support for non-string type keys in `_inject_object` decorator. 

## 0.5.23
- Fix tests distribution under package namespace.

## 0.5.22
- Fix packaging bug for `refit`.

## 0.5.21
- Added pyproject.toml for package.

## 0.5.20
- Added setup.cfg for package.

## 0.5.19
- Update inject function to instantiate with tuple when specified.

## 0.5.18
- Update docs.
- Update the tests to accomodate for pandera latest release `0.18.3`.

## 0.5.17
- Update docs.

## 0.5.16
- Added `validate` decorator.

## 0.5.15
- Remove data fabricator from `config.yml`.

## 0.5.14
- Update markdown files.

## 0.5.13
- Added null check decorator.

## 0.5.12
- Hotfix primary key decorator.

## 0.5.11
- Add input/output name into has_schema and primary key error logging.

## 0.5.10
- Remove `nb-black` from test requirements.

## 0.5.9
- More verbose logging.

## 0.5.8
- Fixed docs due to mermaid support.
## 0.5.7
- Pinned packages to fix test failures.

## 0.5.6
- Added refit style `primary_key` to support input and output primary key checks.

## 0.5.5
- Updated `has_schema` to support return type dictionaries.

## 0.5.4
- Add pytest coverage limits

## 0.5.3
- Added CHANGELOG.md for keeping track of changes to function signatures.

## 0.5.2
- Made release notes consistent with other packages.

## 0.5.1
- Changed the dynamic data generation in myst file to avoid md file changes on each run.

## 0.5.0
- Implemented inline and refit decorators for remove_debug_columns
- Added remove argument to remove_output_debug_columns and remove_input_debug_columns, as a switch to remove or keep
debug columns (default keep)
  - As this is a breaking change, you will need to leverage `remove_output_debug_columns` or
    `remove_input_debug_columns` arguments to revert to old functionality. Use one of the following decorators to
    achieve this:
    ```python
    from refit.v1.core.inline_remove_debug_columns import (
        remove_input_debug_columns,
        remove_output_debug_columns,
    )
    
    @remove_output_debug_columns(remove_output_debug_columns=True)
    def func1(*args, **kwargs):
      ...

    
    @remove_input_debug_columns(remove_input_debug_columns=True)
    def func2(*args, **kwargs):
      ...
    
    # Usage
    func1(args1)
    func2(args2)
    ```
    OR
    ```python
    from refit.v1.core.remove_debug_columns import (
        remove_input_debug_columns,
        remove_output_debug_columns,
    )
    
    @remove_output_debug_columns()
    def func1(*args, **kwargs):
      ...


    @remove_input_debug_columns()
    def func2(*args, **kwargs):
      ...

    # Usage
    func1(args1, remove_output_debug_columns=True)
    func2(args2, remove_input_debug_columns=True)
    ```

## 0.4.0
- Changing defender to inline
  - Changed defender_has_schema to inline_has_schema
  - Changed defender_primary_key to inline_primary_key

## 0.3.16
- Add blacken-docs to format python code in md files.

## 0.3.15
- Rename .myst to .myst.md for better editor support.

## 0.3.14
- Added jupyterbook mechanism. 

## 0.3.13
- Update docs. 

## 0.3.12
- Update docs.

## 0.3.11
- Migrate to new repo and apply test requirements.

## 0.3.10
- Update sorting of imports.

## 0.3.9
- Updated `augment` with the decorator for `_add_input_kwarg_select` .

## 0.3.8
- Fix docstring.

## 0.3.7
- Added upper bound for `scikit-learn` to prevent breaking changes in `1.1.0`.

## 0.3.6
- Remove all upper bounds in requirements. 

## 0.3.5
- Remove all .Rmd files.

## 0.3.4
- Updated internals of primary key check decorators.

## 0.3.3
- Convert .Rmd to .myst.

## 0.3.2
- Add `remove_debug_columns` decorator.

## 0.3.1
- Updated requirements and test requirements.

## 0.3.0
- Updated `unpack_params` from boolean to kwarg to unpack.

## 0.2.1
- Introduced docs folder to place Slack Blast poster.

## 0.2.1
- Introduce `relax` feature to `has_schema` for pandas dataframes.

## 0.2.0
- Rename `input_has_schema` to `has_schema` and add output dataframe check.

## 0.1.38
- Fix Rmd conda env.

## 0.1.37
- Update in test_requirements.in file


## 0.1.36
- Fix doc.


## 0.1.35
- Add `fill_nulls`

## 0.1.34
- Update docs.

## 0.1.33
- Fix `inject` to enable instantiating objects without arguments.

## 0.1.32
- Add `add_input_kwarg_filter`

## 0.1.31
- Make `make_list_regexable` more robust. 

## 0.1.30
- Fix bug in `input_has_schema`.

## 0.1.29
- Add exclude keys functionality to `inject_object()`

## 0.1.28
- Update docs. 

## 0.1.27
- Update `make_list_regexable` to be explicitly turned on.

## 0.1.26

- Update requirements so that `pyspark` is explicitly below `<4.0`
## 0.1.25
- More docs fixes.

## 0.1.24
- Updated `make_regexable` decorator to be able to raise an exception.

## 0.1.23
- Fix docs.

## 0.1.22
- Removed `node` decorator.

## 0.1.21
- Updated individual decorators documentation for `make_regexable` decorator.

## 0.1.20
- Added `augment` decorator.

## 0.1.19
- Update column name in `primary_key` decorator.

## 0.1.18
- Removed useless suppression from `output_primary_key` decorator.

## 0.1.17
- Added `make_list_regexable` decorator.

## 0.1.16
- Cleaned up `requirements.txt`.

## 0.1.15
- Suppressed `consider-using-f-string` pylint message after the version upgrade to `pylint-2.11.1`.

## 0.1.14
- Added `defender primary_key_check` decorator.

## 0.1.13
- Added `output_primary_key` decorator.

## 0.1.12
- Added `add_output_filter` decorator.

## 0.1.11
- Renamed from `requirements.in` to `requirements.txt`.

## 0.1.10
- Updated brix post namespace.

## 0.1.9
- Added `BuiltinFunctionType` check in refit while instantiating object.

## 0.1.8
- Updated pytest fixture scope for `dummy_pd_df` and `dummy_spark_df`.

## 0.1.7
- Added versioning(v1) to refit.

## 0.1.6
- Fix useless pylint disables.

## 0.1.5
- Updated README for individual decorators.

## 0.1.4
- Updated README for individual decorators.

## 0.1.3
- Added wrapper functions and tests for missing decorators.

## 0.1.2
- Added README to showcase individual decorator: `defender` style `has_schema`.

## 0.1.1
- Added individual decorator for `inject_object`.
- Added README to showcase individual decorators: `inject_object`.

## 0.1.0
- Added decorators:
    - `node` which contains `_inject_object`, `_retry`, `_input_has_schema_node`, `_add_input_kwarg_filter` and `_unpack_params`.
    - `input_kwarg_filter`.
    - `input_has_schema`.
    - Defender style `has_schema`.
