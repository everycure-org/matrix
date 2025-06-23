---
title: Documentation
---

If you think there is something missing or that you could improve our documentation, please contribute!

## MKDocs

We use MKDocs for documenting our technology.

For full documentation visit [mkdocs.org](https://www.mkdocs.org). More specifically, we're using `mkdocs material` which is [documented here](https://squidfunk.github.io/mkdocs-material/).

## Visual elements

To add visual elements we provide 2 approaches:

- Embedded Draw.IO drawings: Meant for complex and interactive visualisations
- [MermaidJS](https://mermaid.js.org/syntax): Ability to quickly sketch out simple flows and relationships in code

For Draw.IO we recommend installing the [desktop application](https://formulae.brew.sh/cask/drawio#default), the [vscode plugin](https://marketplace.visualstudio.com/items?itemName=hediet.vscode-drawio) or the [web app](https://app.diagrams.net/).


## Running locally

As with the rest of our projects, we use `uv` for python dependency management. Assuming you have `uv` and `make` installed, you can simply run `make serve` in the docs folder for serving the docs page and developing it.

### Commands

After activating your venv:

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.

### Ordering pages

If you want to control page orders in a subfolder you can place a `.pages` file in the directory and define the order (see root of documentation for example).