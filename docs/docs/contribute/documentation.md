
# MKDocs

We use MKDocs for documenting our technology.

For full documentation visit [mkdocs.org](https://www.mkdocs.org). More specifically, we're using `mkdocs material` which is [documented here](https://squidfunk.github.io/mkdocs-material/).

## Getting started

As with the rest of our projects, we use `uv` for python dependency management. Assuming you have `uv` and `make` installed, you can simply run `make serve` in the docs folder for serving the docs page and developing it.

## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
