# Key files and folders

## Files

### `.env` file

- sets a number of default ENV variables that are needed for the pipeline to run. Note
this file is ignored in the docker image and thus these ENV variables have to set
explicitly for any container to work. This file can also be used to locally run against production data by tweaking the Neo4j or base bucket entries here.

### `Makefile` file

- a file that holds a number of standard operations to be easily executed with `make <command>`. For a tutorial on makefiles check [this link](https://makefiletutorial.com/)

### `requirements.*` files

- define our python requirements. They are compiled through `uv` as can be seen in the `make lock` command.

### `trivy.yaml` file

- used for scanning our docker images for licenses we are wary of and want to avoid using. 

## Folders

### `conf/<env>` folders

- Folders that define which kedro environment is configured how. Check the [kedro documentation](https://docs.kedro.org/en/stable/configuration/configuration_basics.html#configuration-environments) for details on this


### `scripts/` folder

- holds various helper scripts not used in our primary pipeline. 