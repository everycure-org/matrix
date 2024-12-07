# app.py
# source "$(find "$(git rev-parse --show-toplevel)" -type f -name 'activate' -path '*/bin/activate' | head -n 1)"

import os
import polars as pl
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import re
import subprocess
import glob
import scipy.linalg
import multiprocessing as mp
import joblib
import dash_table
import pandas as pd

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

root_path = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
cache_dir = os.path.join(root_path, "apps", "embed_norm", "cached_datasets")


def reduce_dimensions(embeddings, method="tsne", **kwargs):
    if method == "umap":
        return reduce_dimensions_umap(embeddings, **kwargs)
    elif method == "pca":
        return PCA(n_components=2, **kwargs).fit_transform(embeddings)
    elif method == "tsne":
        return TSNE(n_components=2, random_state=42, **kwargs).fit_transform(embeddings)
    else:
        raise ValueError("Unsupported dimensionality reduction method")


def reduce_dimensions_umap(embeddings, n_neighbors=3, min_dist=0.1):
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings


def get_available_options():
    datasets = set()
    models = set()
    categories = set()
    embedding_pattern = re.compile(
        r"^(.+?)(_positive|_negative)?_embeddings_(.+?)_(.+?)(_combinations)?_seed_\d+.*\.pkl$"
    )
    for root, dirs, files in os.walk(os.path.join(cache_dir, "embeddings")):
        for filename in files:
            match = embedding_pattern.match(filename)
            if match:
                dataset_name, sample_type_suffix, category_name, model_name, comb_flag = match.groups()[:5]
                datasets.add(dataset_name)
                models.add(model_name)
                categories.add(category_name)
    return {
        "datasets": sorted(datasets),
        "models": sorted(models),
        "categories": sorted(categories),
    }


options = get_available_options()

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1("Node Embedding Visualization"),
                        html.P("Visualize embeddings from different models and datasets."),
                    ],
                    width=12,
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Dataset"),
                        dcc.Dropdown(
                            id="dataset",
                            options=[{"label": name, "value": name} for name in options["datasets"]],
                            value=options["datasets"][0] if options["datasets"] else None,
                            clearable=False,
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        dbc.Label("Category"),
                        dcc.Dropdown(
                            id="category",
                            options=[{"label": name, "value": name} for name in options["categories"]],
                            value=options["categories"][0] if options["categories"] else None,
                            clearable=False,
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        dbc.Label("Embedding Method"),
                        dcc.Dropdown(
                            id="embedding-method",
                            options=[{"label": name, "value": name} for name in options["models"]],
                            value=options["models"][0] if options["models"] else None,
                            clearable=False,
                        ),
                    ],
                    width=4,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Dimensionality Reduction Method"),
                        dcc.Dropdown(
                            id="dim-reduction-method",
                            options=[
                                {"label": "UMAP", "value": "umap"},
                                {"label": "PCA", "value": "pca"},
                                {"label": "t-SNE", "value": "tsne"},
                            ],
                            value="tsne",
                            clearable=False,
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        html.Div(
                            [
                                dbc.Label("UMAP n_neighbors"),
                                dcc.Slider(
                                    id="n-neighbors",
                                    min=2,
                                    max=50,
                                    step=1,
                                    value=15,
                                    marks={i: str(i) for i in range(2, 51, 5)},
                                ),
                            ],
                            id="umap-n-neighbors-div",
                            style={"display": "block"},
                        )
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        html.Div(
                            [
                                dbc.Label("UMAP min_dist"),
                                dcc.Slider(
                                    id="min-dist",
                                    min=0.0,
                                    max=0.99,
                                    step=0.01,
                                    value=0.1,
                                    marks={i / 10: str(i / 10) for i in range(0, 10)},
                                ),
                            ],
                            id="umap-min-dist-div",
                            style={"display": "block"},
                        )
                    ],
                    width=4,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                dbc.Label("t-SNE Perplexity"),
                                dcc.Slider(
                                    id="perplexity",
                                    min=5,
                                    max=50,
                                    step=1,
                                    value=50,
                                    marks={i: str(i) for i in range(5, 51, 5)},
                                ),
                            ],
                            id="tsne-perplexity-div",
                            style={"display": "none"},
                        )
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        dbc.Label("Point Size"),
                        dcc.Slider(
                            id="point-size",
                            min=5,
                            max=50,
                            step=1,
                            value=10,
                            marks={i: str(i) for i in range(5, 51, 5)},
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        dbc.Label("Apply Clustering"),
                        dbc.Checklist(
                            options=[{"label": "Enable K-Means Clustering", "value": 1}],
                            value=[],
                            id="enable-clustering",
                            switch=True,
                        ),
                    ],
                    width=4,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Number of Clusters"),
                        dcc.Slider(
                            id="num-clusters",
                            min=2,
                            max=10,
                            step=1,
                            value=3,
                            marks={i: str(i) for i in range(2, 11)},
                            disabled=True,
                        ),
                    ],
                    width=4,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Cosine Similarity Threshold"),
                        dcc.Slider(
                            id="cosine-threshold",
                            min=0.0,
                            max=1.0,
                            step=0.001,
                            value=0.96,
                            marks={i / 10: str(i / 10) for i in range(0, 11)},
                        ),
                    ],
                    width=12,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Number of Top Matches"),
                        dcc.Slider(
                            id="num-top-matches",
                            min=1,
                            max=50,
                            step=1,
                            value=10,
                            marks={i: str(i) for i in range(1, 51, 5)},
                        ),
                    ],
                    width=12,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [dcc.Loading(id="loading-embedding-plot", type="default", children=dcc.Graph(id="embedding-plot"))],
                    width=12,
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("Top Positive Matches"),
                        dash_table.DataTable(
                            id="positive-matches-table",
                            columns=[
                                {"name": "ID1", "id": "ID1"},
                                {"name": "ID2", "id": "ID2"},
                                {
                                    "name": "Cosine Similarity",
                                    "id": "Cosine Similarity",
                                    "type": "numeric",
                                    "format": {"specifier": ".4f"},
                                },
                            ],
                            data=[],
                        ),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        html.H2("Top Negative Matches"),
                        dash_table.DataTable(
                            id="negative-matches-table",
                            columns=[
                                {"name": "ID1", "id": "ID1"},
                                {"name": "ID2", "id": "ID2"},
                                {
                                    "name": "Cosine Similarity",
                                    "id": "Cosine Similarity",
                                    "type": "numeric",
                                    "format": {"specifier": ".4f"},
                                },
                            ],
                            data=[],
                        ),
                    ],
                    width=6,
                ),
            ]
        ),
        dcc.Store(id="embedding-data-store"),
    ]
)


@app.callback(
    Output("umap-n-neighbors-div", "style"),
    Output("umap-min-dist-div", "style"),
    Output("tsne-perplexity-div", "style"),
    Input("dim-reduction-method", "value"),
)
def toggle_parameter_controls(dim_reduction_method):
    if dim_reduction_method == "umap":
        umap_style = {"display": "block"}
        tsne_style = {"display": "none"}
    elif dim_reduction_method == "tsne":
        umap_style = {"display": "none"}
        tsne_style = {"display": "block"}
    else:
        umap_style = {"display": "none"}
        tsne_style = {"display": "none"}
    return umap_style, umap_style, tsne_style


@app.callback(Output("num-clusters", "disabled"), Input("enable-clustering", "value"))
def toggle_num_clusters(enable_clustering):
    return not bool(enable_clustering)


def load_seed_data(args):
    emb_file, lbl_file, ids_file, sim_file, sample_type = args
    emb = joblib.load(emb_file)
    lbl = joblib.load(lbl_file)
    ids = joblib.load(ids_file)
    sim = joblib.load(sim_file)
    sample_type_str = "Positive" if sample_type == "_positive" else "Negative"
    return emb, lbl, ids, sample_type_str, sim


@app.callback(
    Output("embedding-data-store", "data"),
    [
        Input("category", "value"),
        Input("dataset", "value"),
        Input("embedding-method", "value"),
    ],
)
def reload_embeddings(category_name, dataset_name, embedding_method):
    embeddings = []
    labels = []
    ids_list = []
    sample_types = []
    similarities = []
    args_list = []
    for sample_type in ["_positive", "_negative"]:
        embeddings_pattern = os.path.join(
            cache_dir,
            "embeddings",
            f"{dataset_name}{sample_type}_embeddings_{category_name}_{embedding_method}_seed_*.pkl",
        )
        labels_pattern = os.path.join(
            cache_dir,
            "embeddings",
            f"{dataset_name}{sample_type}_labels_{category_name}_{embedding_method}_seed_*.pkl",
        )
        ids_pattern = os.path.join(
            cache_dir,
            "embeddings",
            f"{dataset_name}{sample_type}_ids_{category_name}_{embedding_method}_seed_*.pkl",
        )
        sim_pattern = os.path.join(
            cache_dir,
            "embeddings",
            f"{dataset_name}{sample_type}_cosine_similarities_{category_name}_{embedding_method}_seed_*.pkl",
        )
        embedding_files = glob.glob(embeddings_pattern)
        label_files = glob.glob(labels_pattern)
        ids_files = glob.glob(ids_pattern)
        sim_files = glob.glob(sim_pattern)

        def extract_seed(file_path):
            match = re.search(r"seed_(\d+)", file_path)
            return match.group(1) if match else None

        embedding_files_dict = {extract_seed(f): f for f in embedding_files}
        label_files_dict = {extract_seed(f): f for f in label_files}
        ids_files_dict = {extract_seed(f): f for f in ids_files}
        sim_files_dict = {extract_seed(f): f for f in sim_files}
        common_seeds = (
            set(embedding_files_dict.keys())
            & set(label_files_dict.keys())
            & set(ids_files_dict.keys())
            & set(sim_files_dict.keys())
        )
        for seed in common_seeds:
            emb_file = embedding_files_dict[seed]
            lbl_file = label_files_dict[seed]
            ids_file = ids_files_dict[seed]
            sim_file = sim_files_dict[seed]
            args_list.append((emb_file, lbl_file, ids_file, sim_file, sample_type))
    if not args_list:
        return {}
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(load_seed_data, args_list)
    for emb, lbl, ids, sample_type_str, sim in results:
        embeddings.append(emb)
        labels.extend(lbl)
        ids_list.extend(ids)
        sample_types.extend([sample_type_str] * len(lbl))
        similarities.append(sim)
    embeddings = np.vstack(embeddings)
    similarities = scipy.linalg.block_diag(*similarities)
    data_store = {
        "embeddings": embeddings.tolist(),
        "labels": labels,
        "ids": ids_list,
        "sample_type": sample_types,
        "similarities": similarities.tolist(),
    }
    return data_store


reduced_embeddings_cache = {}


@app.callback(
    Output("embedding-plot", "figure"),
    Input("embedding-data-store", "data"),
    Input("dim-reduction-method", "value"),
    Input("n-neighbors", "value"),
    Input("min-dist", "value"),
    Input("perplexity", "value"),
    Input("point-size", "value"),
    Input("enable-clustering", "value"),
    Input("num-clusters", "value"),
    Input("cosine-threshold", "value"),
)
def update_plot(
    embedding_data_store,
    dim_reduction_method,
    n_neighbors,
    min_dist,
    perplexity,
    point_size,
    enable_clustering,
    num_clusters,
    cosine_threshold,
):
    if not embedding_data_store:
        return {}
    embeddings = np.array(embedding_data_store["embeddings"])
    labels = embedding_data_store["labels"]
    sample_types = embedding_data_store["sample_type"]
    similarities = np.array(embedding_data_store["similarities"])
    cache_key = (dim_reduction_method, n_neighbors, min_dist, perplexity)
    if cache_key in reduced_embeddings_cache:
        embeddings_2d = reduced_embeddings_cache[cache_key]
    else:
        if dim_reduction_method == "umap":
            embeddings_2d = reduce_dimensions_umap(embeddings, n_neighbors=n_neighbors, min_dist=min_dist)
        elif dim_reduction_method == "pca":
            embeddings_2d = reduce_dimensions(embeddings, method="pca")
        elif dim_reduction_method == "tsne":
            embeddings_2d = reduce_dimensions(embeddings, method="tsne", perplexity=perplexity)
        else:
            embeddings_2d = embeddings
        reduced_embeddings_cache[cache_key] = embeddings_2d
    df_plot = pl.DataFrame(
        {
            "x": embeddings_2d[:, 0],
            "y": embeddings_2d[:, 1],
            "label": labels,
            "Sample Type": sample_types,
        }
    )
    if enable_clustering:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings_2d)
        df_plot = df_plot.with_column(pl.Series("cluster", clusters))
        color = "cluster"
    else:
        color = "Sample Type"
    df_plot_pd = df_plot.to_pandas()
    if color == "cluster":
        color_values = df_plot_pd["cluster"].tolist()
        marker_color = color_values
        marker_colorscale = "Viridis"
        colorbar = dict(title="Cluster")
    else:
        color_map = {"Positive": "blue", "Negative": "red"}
        marker_color = df_plot_pd["Sample Type"].map(color_map).fillna("gray").tolist()
        marker_colorscale = None
        colorbar = None
    scatter = go.Scattergl(
        x=df_plot_pd["x"],
        y=df_plot_pd["y"],
        mode="markers",
        marker=dict(
            size=point_size,
            color=marker_color,
            colorscale=marker_colorscale,
            showscale=bool(marker_colorscale),
            colorbar=colorbar,
            line=dict(width=0),
            opacity=0.8,
        ),
        text=df_plot_pd["label"],
        hovertemplate="%{text}",
        customdata=df_plot_pd["label"],
    )
    fig = go.Figure(data=[scatter])
    indices = np.where(similarities >= cosine_threshold)
    x0 = embeddings_2d[indices[0], 0]
    y0 = embeddings_2d[indices[0], 1]
    x1 = embeddings_2d[indices[1], 0]
    y1 = embeddings_2d[indices[1], 1]
    lines_x = []
    lines_y = []
    for xi0, yi0, xi1, yi1 in zip(x0, y0, x1, y1):
        if xi0 == xi1 and yi0 == yi1:
            continue
        lines_x.extend([xi0, xi1, None])
        lines_y.extend([yi0, yi1, None])
    fig.add_trace(
        go.Scattergl(
            x=lines_x,
            y=lines_y,
            mode="lines",
            line=dict(color="gray", width=0.5),
            hoverinfo="none",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Embeddings Visualization",
        hovermode="closest",
        hoverlabel=dict(align="left", font_size=12, bgcolor="rgba(255, 255, 255, 0.8)"),
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


@app.callback(
    [
        Output("positive-matches-table", "data"),
        Output("negative-matches-table", "data"),
    ],
    [
        Input("embedding-data-store", "data"),
        Input("cosine-threshold", "value"),
        Input("num-top-matches", "value"),
    ],
)
def update_tables(embedding_data_store, cosine_threshold, num_top_matches):
    if not embedding_data_store:
        return [], []
    embeddings = np.array(embedding_data_store["embeddings"])
    ids = embedding_data_store["ids"]
    sample_types = embedding_data_store["sample_type"]
    positive_indices = [i for i, st in enumerate(sample_types) if st == "Positive"]
    negative_indices = [i for i, st in enumerate(sample_types) if st == "Negative"]
    positive_embeddings = embeddings[positive_indices]
    positive_ids = [ids[i] for i in positive_indices]
    negative_embeddings = embeddings[negative_indices]
    negative_ids = [ids[i] for i in negative_indices]
    positive_similarities = cosine_similarity(positive_embeddings)
    pos_upper_tri_indices = np.triu_indices(len(positive_embeddings), k=1)
    pos_sim_values = positive_similarities[pos_upper_tri_indices]
    pos_mask = pos_sim_values >= cosine_threshold
    pos_i_indices = pos_upper_tri_indices[0][pos_mask]
    pos_j_indices = pos_upper_tri_indices[1][pos_mask]
    pos_similarities_selected = pos_sim_values[pos_mask]
    positive_matches = pd.DataFrame(
        {
            "ID1": [positive_ids[i] for i in pos_i_indices],
            "ID2": [positive_ids[j] for j in pos_j_indices],
            "Cosine Similarity": pos_similarities_selected,
        }
    )
    negative_similarities = cosine_similarity(negative_embeddings)
    neg_upper_tri_indices = np.triu_indices(len(negative_embeddings), k=1)
    neg_sim_values = negative_similarities[neg_upper_tri_indices]
    neg_mask = neg_sim_values >= cosine_threshold
    neg_i_indices = neg_upper_tri_indices[0][neg_mask]
    neg_j_indices = neg_upper_tri_indices[1][neg_mask]
    neg_similarities_selected = neg_sim_values[neg_mask]
    negative_matches = pd.DataFrame(
        {
            "ID1": [negative_ids[i] for i in neg_i_indices],
            "ID2": [negative_ids[j] for j in neg_j_indices],
            "Cosine Similarity": neg_similarities_selected,
        }
    )
    N = num_top_matches
    positive_matches = positive_matches.sort_values(by="Cosine Similarity", ascending=False).head(N)
    negative_matches = negative_matches.sort_values(by="Cosine Similarity", ascending=False).head(N)
    positive_matches_dict = positive_matches.to_dict("records")
    negative_matches_dict = negative_matches.to_dict("records")
    return positive_matches_dict, negative_matches_dict


if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False, host="0.0.0.0", port=3000)
