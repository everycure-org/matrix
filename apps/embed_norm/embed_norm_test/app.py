# app.py

import os
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

seed1 = 54321  # Seed for positive samples
seed2 = 67890  # Seed for negative samples
cache_dir = "cached_datasets"


def reduce_dimensions(embeddings, method="umap", **kwargs):
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


def get_available_datasets_models_and_datasets():
    datasets = set()
    models = set()
    categories = set()
    embedding_types = set()
    embedding_pattern = re.compile(r"^(.+?)_embeddings_(.+?)_(.+?)(_combinations)?_seed_\d+\.pkl$")
    for filename in os.listdir(cache_dir):
        match = embedding_pattern.match(filename)
        if match:
            dataset_name, category_name, model_name, comb_flag = match.groups()
            embedding_type = "Combination" if comb_flag == "_combinations" else "Standard"
            datasets.add(dataset_name)
            models.add(model_name)
            categories.add(category_name)
            embedding_types.add(embedding_type)
    return sorted(datasets), sorted(models), sorted(categories), sorted(embedding_types)


datasets_list, embedding_models, categories_list, embedding_types = get_available_datasets_models_and_datasets()

embedding_options = [{"label": name, "value": name} for name in embedding_models]
dataset_options = [{"label": name, "value": name} for name in datasets_list]
category_options = [{"label": name, "value": name} for name in categories_list]
embedding_type_options = [{"label": etype, "value": etype} for etype in embedding_types]

app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1("Node Embedding Visualization"),
                        html.P("Current text embedding: all_names + all_categories"),
                    ],
                    width=16,
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
                            options=dataset_options,
                            value=datasets_list[0] if datasets_list else None,
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
                            options=category_options,
                            value=categories_list[0] if categories_list else None,
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
                            options=embedding_options,
                            value=embedding_models[0] if embedding_models else None,
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
                        dbc.Label("Embedding Type"),
                        dcc.Dropdown(
                            id="embedding-type", options=embedding_type_options, value="Standard", clearable=False
                        ),
                    ],
                    width=6,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("Sample Type"),
                        dcc.Dropdown(
                            id="sample-type",
                            options=[
                                {"label": "Both Positive and Negative", "value": "both"},
                                {"label": "Positive Only", "value": "positive"},
                                {"label": "Negative Only", "value": "negative"},
                            ],
                            value="both",
                            clearable=False,
                        ),
                    ],
                    width=6,
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
                            value="umap",
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
                                    value=3,
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
                                    value=30,
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
                            id="point-size", min=5, max=50, step=1, value=10, marks={i: str(i) for i in range(5, 51, 5)}
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
                            step=0.01,
                            value=0.8,
                            marks={i / 10: str(i / 10) for i in range(0, 11)},
                        ),
                    ],
                    width=4,
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


@app.callback(
    Output("embedding-data-store", "data"),
    [
        Input("category", "value"),
        Input("dataset", "value"),
        Input("embedding-method", "value"),
        Input("embedding-type", "value"),
        Input("sample-type", "value"),
    ],
)
def reload_embeddings(category_name, dataset_name, embedding_method, embedding_type, sample_type):
    embeddings_files = {}
    labels_files = {}
    similarities_files = {}

    embedding_suffix = "_combinations" if embedding_type == "Combination" else ""

    for s_type, seed in [("positive", seed1), ("negative", seed2)]:
        embeddings_files[s_type] = os.path.join(
            cache_dir, f"{dataset_name}_embeddings_{category_name}_{embedding_method}{embedding_suffix}_seed_{seed}.pkl"
        )
        labels_files[s_type] = os.path.join(
            cache_dir, f"{dataset_name}_labels_{category_name}_{embedding_method}{embedding_suffix}_seed_{seed}.pkl"
        )
        similarities_files[s_type] = os.path.join(
            cache_dir,
            f"{dataset_name}_cosine_similarities_{category_name}_{embedding_method}{embedding_suffix}_seed_{seed}.pkl",
        )

    embeddings_list = []
    labels_list = []
    sample_types = []
    similarities_list = []

    sample_types_to_load = []
    if sample_type == "both":
        sample_types_to_load = ["positive", "negative"]
    else:
        sample_types_to_load = [sample_type]

    for s_type in sample_types_to_load:
        embeddings_file = embeddings_files[s_type]
        labels_file = labels_files[s_type]
        sim_file = similarities_files[s_type]
        if not os.path.exists(embeddings_file) or not os.path.exists(labels_file) or not os.path.exists(sim_file):
            continue
        with open(embeddings_file, "rb") as f:
            embeddings = pickle.load(f)
        with open(labels_file, "rb") as f:
            labels = pickle.load(f)
        with open(sim_file, "rb") as f:
            similarities = pickle.load(f)
        embeddings_list.append(embeddings)
        labels_list.extend(labels)
        sample_types.extend([s_type.capitalize()] * len(labels))
        similarities_list.append(similarities)

    if not embeddings_list:
        return {}

    embeddings_combined = np.vstack(embeddings_list)
    similarities_combined = None
    if len(similarities_list) == 1:
        similarities_combined = similarities_list[0]
    else:
        similarities_combined = cosine_similarity(embeddings_combined)

    data_store = {
        "embeddings_combined": embeddings_combined.tolist(),
        "labels_list": labels_list,
        "sample_types": sample_types,
        "similarities_combined": similarities_combined.tolist(),
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

    embeddings_combined = np.array(embedding_data_store["embeddings_combined"])
    labels_list = embedding_data_store["labels_list"]
    sample_types = embedding_data_store["sample_types"]
    similarities_combined = np.array(embedding_data_store["similarities_combined"])

    cache_key = (dim_reduction_method, n_neighbors, min_dist, perplexity)
    if cache_key in reduced_embeddings_cache:
        embeddings_2d = reduced_embeddings_cache[cache_key]
    else:
        if dim_reduction_method == "umap":
            embeddings_2d = reduce_dimensions_umap(embeddings_combined, n_neighbors=n_neighbors, min_dist=min_dist)
        elif dim_reduction_method == "pca":
            embeddings_2d = reduce_dimensions(embeddings_combined, method="pca")
        elif dim_reduction_method == "tsne":
            embeddings_2d = reduce_dimensions(embeddings_combined, method="tsne", perplexity=perplexity)
        else:
            embeddings_2d = embeddings_combined
        reduced_embeddings_cache[cache_key] = embeddings_2d

    if embeddings_combined.shape[0] != len(labels_list) or embeddings_combined.shape[0] != len(sample_types):
        return {}

    df_plot = pd.DataFrame(
        {"x": embeddings_2d[:, 0], "y": embeddings_2d[:, 1], "label": labels_list, "Sample Type": sample_types}
    )

    if enable_clustering:
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df_plot["cluster"] = kmeans.fit_predict(embeddings_2d)
        color = "cluster"
    else:
        color = "Sample Type"

    fig = px.scatter(
        df_plot, x="x", y="y", hover_data=None, custom_data=["label"], color=color, labels={"x": "", "y": ""}
    )
    fig.update_traces(hovertemplate="%{customdata[0]}", marker=dict(size=point_size))
    fig.update_layout(
        title="Embeddings Visualization",
        hovermode="closest",
        hoverlabel=dict(align="left", font_size=12, bgcolor="rgba(255, 255, 255, 0.2)"),
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
    )

    # Add lines for cosine similarity
    indices = np.where(similarities_combined >= cosine_threshold)
    x0 = embeddings_2d[indices[0], 0]
    y0 = embeddings_2d[indices[0], 1]
    x1 = embeddings_2d[indices[1], 0]
    y1 = embeddings_2d[indices[1], 1]
    lines = []
    for xi0, yi0, xi1, yi1 in zip(x0, y0, x1, y1):
        if xi0 == xi1 and yi0 == yi1:
            continue
        lines.append(dict(type="line", x0=xi0, y0=yi0, x1=xi1, y1=yi1, line=dict(color="gray", width=0.5)))
    fig.update_layout(shapes=lines)

    return fig


if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False, host="0.0.0.0", port=3000)
