# MATRIX IO

## Installing

```shell
make install
```

## Running Fabricator

### Create KG Schema Snapshot
This produces a json output that can be reviewed/edited and then included in a subsequent 'build-yaml-from-kg-schema-snapshot' command.
```shell
fabricator create-kg-schema-snapshot --nodes <nodes_file> --edges <edges_file> -o schema-snapshot.json
```

### Build YAML From KG Schema Snapshot
This produces an output that can be a drop-in yaml intended for Fabricator use.
```shell
fabricator build-yaml-from-kg-schema-snapshot --nodes <nodes_file> --edges <edges_file> -o <fabricator_yaml_output> -s schema-snapshot.json
```

### Build YAML From KGX
This produces an output that can be a drop-in yaml intended for Fabricator use.  Note that this does not have the intermediary step of reviewing/filtering columms.
```shell
fabricator build-yaml-from-kgx --nodes <nodes_file> --edges <edges_file> -o <fabricator_yaml_output>
```
