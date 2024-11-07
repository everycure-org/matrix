# Key files and folders

## Files

### Environment files

- `.env.defaults`: Contains shared default environment variables and documentation for all available configuration options. This file is version controlled and serves both as default configuration and as documentation.
- `.env`: Sets local environment variables and credentials, overriding defaults as needed. This file is gitignored to protect sensitive information.

### `Makefile` file

- a file that holds a number of standard operations to be easily executed with `make <command>`. For a tutorial on makefiles check [this link](https://makefiletutorial.com/)

### `requirements.*` files

- define our python requirements. They are compiled through `uv` as can be seen in the `make lock` command.
<!--

### `trivy.yaml` file

- used for scanning our docker images for licenses we are wary of and want to avoid using. 
-->

## Folders

### `conf/<env>` folders

- Folders that define which kedro environment is configured how. Check the [kedro documentation](https://docs.kedro.org/en/stable/configuration/configuration_basics.html#configuration-environments) for details on this


### `scripts/` folder

- holds various helper scripts not used in our primary pipeline. 