# MOA Extraction

This is a proof of concept for a Streamlit app that allows users to explore 
MOA predictions for a given drug-disease pair.

## Setup

### Set up your env and install dependencies
```bash
uv pip install -r requirements.txt
```

### Check makefile
Check the makefile to make sure the paths are correct for your local env.

### Run the app
```bash
# Pull data from GCS and build local db  
make data
# Run the app
make run
```

Bump the version in the 'version' file when merging.