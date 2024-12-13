# %%
# -------------------------------------------------------------------------------------------------
# SETUP
# -------------------------------------------------------------------------------------------------
# shut up mlflow
import mlflow
try:    
    mlflow.end_run()
except:
    pass

import pyspark as ps
import os
import pandas as pd
from pathlib import Path
import subprocess
import pyspark.sql.functions as f
from pyspark.sql import DataFrame
# import spark 
%load_ext autoreload
%autoreload 2
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
console = Console()
from bmt import toolkit
tk =toolkit.Toolkit()

# hack that moves this notebook context into the kedro path
root_path = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode().strip()
os.chdir(Path(root_path) / 'pipelines' / 'matrix')
#import sys
#sys.path.append(Path(root_path) / 'pipelines' / 'matrix' / 'tests')

# this loads various objects into the context, see 
# https://docs.kedro.org/en/stable/notebooks_and_ipython/kedro_and_notebooks.html#kedro-line-magics
%load_ext kedro.ipython
# os.environ["NEO4J_PASSWORD"] = "changeme-123-safe!"
# %reload_kedro  
# %reload_kedro  --env test
%reload_kedro

# %%
edges = catalog.load("ingestion.int.rtx_kg2.edges")
gt = catalog.load("modelling.raw.ground_truth.positives@spark")
print(f"gt count: {gt.count()}")
print(f"edges count: {edges.count()}")
# %%
# -------------------------------------------------------------------------------------------------
# CORE LOGIC
# -------------------------------------------------------------------------------------------------
def get_undirected_graph(df):
    return df.withColumnsRenamed({
        "subject": "o",
        "object": "s"
    }).withColumnsRenamed({
        "o": "object",
        "s": "subject"
    }).union(df)
# %%
# %%
# 1. undirect the graph by creating 2 edges for each edge
edges_un = get_undirected_graph(edges).select("subject", "object")

# 2. start building the 2 hop paths starting from the source

def join_next_hop(df, edges_df, hop, reverse=False):
    diff = -1 if reverse else 1
    # log some join information
    edges = edges_df.withColumnsRenamed({
        "subject": f"node_{hop-diff}",
        "object": f"node_{hop}"
    }).alias("edges")
    left = df.alias("left")
    return left.join(edges, on=f"node_{hop-diff}", how="inner")


def meet_in_the_middle(df, edges_df, hops):
    assert hops % 2 == 0, "hops must be even"
    # FUTURE handle uneven hops by meeting not perfectly in the middle
    half = hops // 2

    # walk the left side
    left = df.withColumn("node_0", df.source).select("node_0")
    for i in range(1, half+1):
        left = left.transform(join_next_hop, edges_df, i)

    # walk the right side
    right = df.withColumn(f"node_{hops}", df.target).select(f"node_{hops}")
    for i in range(hops, half-1, -1):
        right = right.transform(join_next_hop, edges_df, i-1, reverse=True)

    # # join the two sides
    return left.join(right, on=f"node_{half}", how="inner")

# -------------------------------------------------------------------------------------------------
# BENCHMARKING
# -------------------------------------------------------------------------------------------------
def benchmark_runtime_for_pair_counts(gt, edges_un, hops, pair_count: int):
    gt = gt.limit(pair_count)
    start = time.time()
    paths = meet_in_the_middle(gt, edges_un, hops)
    count = paths.count()
    end = time.time()
    print(f"found {count} paths for {pair_count} pairs and {hops} hops")
    print(f"time taken: {time.time() - start} seconds")

import time
for i in range(1000, 15000, 3000):
    #benchmark_runtime_for_pair_counts(gt, edges_un, hops=2, pair_count=i)
    benchmark_runtime_for_pair_counts(gt, edges_un, hops=4, pair_count=i)
# %%