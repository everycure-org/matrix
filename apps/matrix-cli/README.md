# MATRIX IO

## Installing

```shell
make install
```

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

