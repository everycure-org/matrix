# MATRIX IO

## Installing

```shell
make install
```

## Running Fabricator

### Create a Knowledge Graph schema snapshot in JSON
This script creates a KG schema in json format.

Note: Use this if you would like to revise the schema that would then be used in a subsequent step ("Build YAML from a Knowledge Graph schema snapshot").
```shell
fabricator create-kg-schema-snapshot --nodes <kgx_nodes_file> --edges <kgx_edges_file> --output <output_file>
```

### Build YAML from a Knowledge Graph schema snapshot
This script builds a yaml from a KG schema and KGX nodes and edges files. 

Note: Uses the schema-snapshot JSON from the prerequisite step ("Create a Knowledge Graph schema snapshot in JSON").
```shell
fabricator build-yaml-from-kg-schema-snapshot --nodes <kgx_nodes_file> --edges <kgx_edges_file> --schema-snapshot <output_from_prerequisite_step> --output <output_file>
```

### Build YAML from KGX
This script bypasses any steps to revise columns and directly generates yaml based on the KGX nodes and edges files. 
```shell
fabricator build_yaml_from_kgx --nodes <kgx_nodes_file> --edges <kgx_edges_file> --output <output_file>

## Running PrimeKG

### Building a KGX Edges file from PrimeKG
This script transforms the PrimeKG file into a Biolink compatible KGX edges file.
```shell
primekg build-edges -i <primekg_kg_csv> --output edges.tsv
```

### Building a KGX Nodes file from PrimeKG
This script transforms the PrimeKG file into a Biolink compatible KGX nodes file.
```shell
primekg build-nodes -a <primekg_drug_features_csv> -b <primekg_disease_features_csv> -n <primekg_nodes_csv> --output nodes.tsv
```

### Print Biolink Predicate Mappings
This produces a json output that can be reviewed/edited and then included in a subsequent 'build-yaml-from-kg-schema-snapshot' command.
```shell
primekg print-predicate-mappings --nodes <nodes_file> --edges <edges_file>
```

