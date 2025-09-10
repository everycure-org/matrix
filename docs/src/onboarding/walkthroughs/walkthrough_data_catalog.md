# Walkthrough: Data Catalog

This is an in-depth exploration of the Data Catalog, explaining its structure and how it works.

It is not meant as an exhaustive explanation of every file, but it is a companion to the [Walkthrough: Data Scientist Daniel](walkthrough.md) that explains how the data catalog is used in the code.


## Anchor usage

```yaml
_pandas_csv: &_pandas_csv
  type:  pandas.CSVDataset

_spark_parquet_ds: &_spark_parquet
  type: matrix.datasets.gcp.LazySparkDataset
  file_format: parquet
  save_args:
    mode: overwrite

_spark_csv: &_spark_csv
  type: spark.SparkDataset
  file_format: csv
  load_args:
    header: True
  save_args:
    mode: overwrite

_layer_raw: &_layer_raw
  metadata:
    kedro-viz:
      layer: raw

_layer_int: &_layer_int
  metadata:
    kedro-viz:
      layer: integration
```

**_pandas_csv: &_pandas_csv**

- This defines a dataset configuration using the pandas.CSVDataset class, which is a standard Kedro dataset for handling CSV files.
- The anchor (`&_pandas_csv`) allows this configuration to be reused elsewhere in the catalog by referencing it with `*_pandas_csv`.

**_spark_parquet_ds: &_spark_parquet**

- This defines a configuration for a Spark dataset using the `matrix.datasets.gcp.LazySparkDataset` class.
- The file_format is set to parquet, indicating that Parquet files are used.
- The save_args section specifies that files should be saved in “overwrite” mode, which means any existing data will be replaced when writing.

**_layer_raw: &_layer_raw**

- This section is defining metadata for the raw layer of the pipeline. The kedro-viz tag is used to organize datasets into layers, which is useful for visualization in Kedro-Viz.
- The alias _layer_raw can be referenced in other parts of the catalog to apply this layer metadata.

 
## Ingestion resources

```yaml
# -------------------------------------------------------------------------
# RTX-KG2
# -------------------------------------------------------------------------  

ingestion.raw.rtx_kg2.nodes@pandas:
  <<: [*_pandas_csv, *_layer_raw]
  filepath: ${globals:paths.raw}/rtx_kg2/${globals:data_sources.rtx_kg2.version}/nodes_c.tsv
  load_args:
    sep: "\t"
  save_args:
    header: false
    index: false
    sep: "\t"
```

- `ingestion.raw.rtx_kg2.nodes` dataset in pandas (`pandas.CSVDataset`) format is defined here.
- `<<: [*_pandas_csv, *_layer_raw]`
    - This line indicates the use of YAML merge syntax, which allows the configuration to inherit properties from the anchors _pandas_csv and _layer_raw:
        - `_pandas_csv` defines the use of the pandas.CSVDataset, so this dataset will use Pandas for reading and writing.
        - `_layer_raw` includes metadata specifying that this dataset belongs to the “raw” layer for visualization purposes.
    - By merging these, you avoid redefining the type: pandas.CSVDataset and the layer metadata here.
- `filepath: ${globals:paths.raw}/rtx_kg2/${globals:data_sources.rtx_kg2.version}/nodes_c.tsv`
    - The filepath points to the location of the file that this dataset will load or save.
    - It uses variables from the globals section of the catalog.yml file, which is a common Kedro practice to centralize configuration values:
        - `${globals:paths.raw}` refers to the base path for raw data files.
        - `${globals:data_sources.rtx_kg2.version}` dynamically pulls the version of the rtx_kg2 data source.
    - The full path likely resolves to something like `path_to_raw_folder/rtx_kg2/version_xxx/nodes_c.tsv`
- `load_args`:
    - sep: `"\t"` indicates that the file being loaded is a tab-separated values (TSV) file.
    - These are the custom arguments passed to Pandas when loading the dataset, as TSV files use tab characters (`\t`) as the separator.
- `save_args`:
    - Specifies how the file should be saved back to disk:
        - `header`: false means the header row won’t be written to the file when saving.
        - `index`: false ensures that the index (row numbers) is not included in the saved file.
        - `sep`: `"\t"` confirms that the file will be saved as a TSV (tab-separated values) file.

```yaml
ingestion.raw.rtx_kg2.nodes@spark:
  <<: *_layer_raw
  type: matrix.datasets.gcp.SparkWithSchemaDataset
  filepath: ${globals:paths.raw}/rtx_kg2/${globals:data_sources.rtx_kg2.version}/nodes_c.tsv
  file_format: csv
  load_args:
    sep: "\t"
    header: false
    index: false
    schema:
      object: pyspark.sql.types.StructType
      fields:
        - object: pyspark.sql.types.StructField
          name: id
          dataType: 
            object: pyspark.sql.types.StringType
          nullable: False

	      ... more schema fields follow
	      
	      
ingestion.raw.rtx_kg2.edges@pandas:
  <<: [*_pandas_csv, *_layer_raw]
  filepath: ${globals:paths.raw}/rtx_kg2/${globals:data_sources.rtx_kg2.version}/edges_c.tsv
	...
	
ingestion.raw.rtx_kg2.edges@spark:
  <<: *_layer_raw
  type: matrix.datasets.gcp.SparkWithSchemaDataset
  filepath: ${globals:paths.raw}/rtx_kg2/${globals:data_sources.rtx_kg2.version}/edges_c.tsv
	...
	
ingestion.int.rtx_kg2.nodes:
  <<: [*_spark_parquet, *_layer_int]
  filepath: ${globals:paths.integration}/rtx_kg2/nodes

ingestion.int.rtx_kg2.edges:
  <<: [*_spark_parquet, *_layer_int]
  filepath: ${globals:paths.integration}/rtx_kg2/edges
```