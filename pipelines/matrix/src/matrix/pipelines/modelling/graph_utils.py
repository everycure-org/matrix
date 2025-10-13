from __future__ import annotations

from typing import Dict, Hashable

import numpy as np
import pandas as pd


def _aggregate_neighbors(uu: np.ndarray, vv: np.ndarray) -> Dict[Hashable, np.ndarray]:
    df = pd.DataFrame({"u": uu, "v": vv})
    neighbors: Dict[Hashable, np.ndarray] = {}
    for u, grp in df.groupby("u"):  # noqa: PD011
        arr = grp["v"].to_numpy()
        if len(arr):
            arr = np.unique(arr)
        neighbors[u] = arr
    return neighbors


essential_train_cols = {"source", "target", "split", "y"}


def build_neighbor_map_from_edges(
    edges: pd.DataFrame,
    *,
    source_col: str = "source",
    target_col: str = "target",
    undirected: bool = True,
    drop_self_loops: bool = True,
) -> Dict[Hashable, np.ndarray]:
    """
    Build a neighbor map (node -> sorted unique neighbor ids) from an edges DataFrame.

    IMPORTANT: To avoid leakage when used for feature computation (CN/Jaccard/AA/RA),
    ensure `edges` only contains TRAIN positives for the current split/fold.

    Args:
        edges: Pandas DataFrame with at least [source_col, target_col]. Should contain ONLY
               training positives for the split to prevent leakage.
        source_col: Name of source column.
        target_col: Name of target column.
        undirected: If True, add both (u->v) and (v->u).
        drop_self_loops: If True, remove u->u entries if present.

    Returns:
        Dict mapping node id -> sorted numpy array of unique neighbor ids.
    """
    if edges is None or edges.empty:
        return {}

    s = edges[source_col].to_numpy()
    t = edges[target_col].to_numpy()

    if undirected:
        uu = np.concatenate([s, t])
        vv = np.concatenate([t, s])
    else:
        uu, vv = s, t

    if drop_self_loops:
        mask = uu != vv
        uu = uu[mask]
        vv = vv[mask]

    return _aggregate_neighbors(uu, vv)


def build_neighbor_map_from_train(
    edges: pd.DataFrame,
    *,
    source_col: str = "source",
    target_col: str = "target",
    split_col: str = "split",
    train_value: str = "TRAIN",
    label_col: str = "y",
    pos_label: int | str = 1,
    undirected: bool = True,
    drop_self_loops: bool = True,
) -> Dict[Hashable, np.ndarray]:
    """
    Build a leakage-safe neighbor map using only TRAIN positives from the given edges dataframe.

    - Filters to rows where split == train_value and y == pos_label
    - Builds undirected adjacency by default (adds both (u->v) and (v->u))
    - Removes self-loops if requested

    Args:
        edges: Pandas DataFrame with at least [source_col, target_col, split_col, label_col]
        source_col: Name of source column.
        target_col: Name of target column.
        split_col: Column denoting split (e.g., TRAIN/TEST)
        train_value: Value in split_col that denotes training rows (default: "TRAIN")
        label_col: Column holding labels; positive edges used for adjacency
        pos_label: The positive class label value (default: 1)
        undirected: If True, add both (u->v) and (v->u)
        drop_self_loops: If True, drop u==v pairs

    Returns:
        Dict mapping node id -> sorted numpy array of unique neighbor ids.
    """
    if edges is None or edges.empty:
        return {}

    # Filter to TRAIN positives only to avoid leakage into validation/test
    mask = (edges[split_col] == train_value) & (edges[label_col] == pos_label)
    e = edges.loc[mask, [source_col, target_col]]
    if e.empty:
        return {}

    s = e[source_col].to_numpy()
    t = e[target_col].to_numpy()

    if undirected:
        uu = np.concatenate([s, t])
        vv = np.concatenate([t, s])
    else:
        uu, vv = s, t

    if drop_self_loops:
        good = uu != vv
        uu = uu[good]
        vv = vv[good]

    return _aggregate_neighbors(uu, vv)
