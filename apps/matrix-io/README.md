# MATRIX IO

## Installing

```shell
make install
```

## Running PrimeKG

### Print Biolink Predicate Mappings
This produces a json output that can be reviewed/edited and then included in a subsequent 'build-yaml-from-kg-schema-snapshot' command.
```shell
primekg print-predicate-mappings --nodes <nodes_file> --edges <edges_file>
```

