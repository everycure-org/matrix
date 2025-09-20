# MATRIX Kedro Project

This directory is a codebase for data processing and ML modelling for our drug repurposing platform. Our pipeline is built using [Kedro documentation](https://docs.kedro.org) so we encourage you to get familiar with this library to utilize kedro to its full potential within MATRIX

## How to install dependencies & run the pipeline? 
More detailed instructions on setting up your MATRIX project can be found here however below we also provide quick & simple instructions for Mac OS (instructions for other OS can be found within documentation).

0. Install dependencies required for our pipeline
```bash
# Install pyenv - for python version management
brew install pyenv 
pyenv install 3.11 

# Install uv - for library management
brew install uv python@3.11

# Install docker - required for Neo4j set-up
brew install --cask docker #installs docker desktop
brew install docker docker-compose #installs CLI commands

# Install Java - required for PySpark operations
brew install openjdk@17
brew link --overwrite openjdk@17
```

1. Create Virtual Enviroment & Install Dependencies
```
# From the root directory
make install
```

Declare any dependencies in `pyproject.toml`

To install them, run the following within your virtual environment (we recommend using [uv](https://docs.astral.sh/uv/) & Python 3.11):

```
make install
```


## Rules and guidelines

In order to get the best out of the kedro project template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a [data engineering convention](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention)
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`
## How to test your Kedro project

Have a look at the files `src/tests/test_run.py` and `src/tests/pipelines/data_science/test_pipeline.py` for instructions on how to write your tests. Run the tests as follows:

```
pytest
```

To configure the coverage threshold, look at the `.coveragerc` file.

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `catalog`, `context`, `pipelines` and `session`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `make install` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
uv pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
uv pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

[Further information about using notebooks for experiments within Kedro projects](https://docs.kedro.org/en/develop/notebooks_and_ipython/kedro_and_notebooks.html).

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html).
