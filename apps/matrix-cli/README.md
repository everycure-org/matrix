# Matrix CLI

Command-line interface for various tasks in the MATRIX project. This tool
provides utilities for managing GitHub operations, releases, and code analysis.
It's not yet fully developed and mostly just meant to bundle convenience
commands.

## Features

Currently supports:

- GitHub team management (`gh-users`)
- Release management and notes generation (`releases`)
- Code analysis and summarization (`code`)

## Installation

Ensure you have uv installed, then run the following command in the CLI root:

```bash
uv sync
```

## Prerequisites

1. **Google Cloud Authentication**
   
   For AI-powered features (code summarization, release notes), authenticate with Google Cloud:

   ```bash
   gcloud auth login
   gcloud auth application-default login
   ```

2. **GitHub CLI**
   
   For GitHub-related operations, ensure you have the GitHub CLI installed and authenticated.

   ```bash
   gh auth login
   ```

## Usage

Please call the matrix CLI itself and see the help text for each command.

```bash
# assumes you have activated the virtual environment
matrix
```

## Development Status

This CLI is in early development. Current functionality focuses on basic GitHub
operations and release management, with plans to expand capabilities in future
releases.

## Contributing

Contributions are welcome! 
