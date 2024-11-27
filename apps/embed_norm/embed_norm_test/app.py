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
import re

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

cache_dir = "../cached_datasets"


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


def get_available_options():
    datasets = set()
    models = set()
    categories = set()
    embedding_types = set()
    sample_types = set()
    seeds = set()
    embedding_pattern = re.compile(r"^(.+?)_embeddings_(.+?)_(.+?)(_combinations)?_seed_(\d+)(.+?)?\.pkl$")

    for root, dirs, files in os.walk(cache_dir):
        for filename in files:
            match = embedding_pattern.match(filename)
            if match:
                # Extract groups from the regex match
                dataset_name, category_name, model_name, comb_flag, seed, cache_suffix = match.groups()

                # Determine the embedding type based on comb_flag
                embedding_type = "Combination" if comb_flag == "_combinations" else "Standard"

                # Extract sample_type using another regex search
                sample_type_match = re.search(r"_sample_(.+)", cache_suffix or "")
                sample_type = sample_type_match.group(1) if sample_type_match else "unknown"

                # Add the extracted information to respective sets
                datasets.add(dataset_name)
                models.add(model_name)
                categories.add(category_name)
                embedding_types.add(embedding_type)
                sample_types.add(sample_type)
                seeds.add(seed)

    # for filename in os.listdir(cache_dir):
    #     match = embedding_pattern.match(filename)
    #     if match:
    #         dataset_name, category_name, model_name, comb_flag, seed, cache_suffix = match.groups()
    #         embedding_type = "Combination" if comb_flag == "_combinations" else "Standard"
    #         sample_type_match = re.search(r"_sample_(.+)", cache_suffix or "")
    #         sample_type = sample_type_match.group(1) if sample_type_match else "unknown"
    #         datasets.add(dataset_name)
    #         models.add(model_name)
    #         categories.add(category_name)
    #         embedding_types.add(embedding_type)
    #         sample_types.add(sample_type)
    #         seeds.add(seed)
    return {
        "datasets": sorted(datasets),
        "models": sorted(models),
        "categories": sorted(categories),
        "embedding_types": sorted(embedding_types),
        "sample_types": sorted(sample_types),
        "seeds": sorted(seeds),
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
                        dbc.Label("Embedding Type"),
                        dcc.Dropdown(
                            id="embedding-type",
                            options=[{"label": name, "value": name} for name in options["embedding_types"]],
                            value="Standard",
                            clearable=False,
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        dbc.Label("Sample Type"),
                        dcc.Dropdown(
                            id="sample-type",
                            options=[{"label": name, "value": name} for name in options["sample_types"]],
                            value=options["sample_types"][0] if options["sample_types"] else None,
                            clearable=False,
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        dbc.Label("Seed"),
                        dcc.Dropdown(
                            id="seed",
                            options=[{"label": name, "value": name} for name in options["seeds"]],
                            value=options["seeds"][0] if options["seeds"] else None,
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
        Input("seed", "value"),
    ],
)
def reload_embeddings(category_name, dataset_name, embedding_method, embedding_type, sample_type, seed):
    embeddings_file = os.path.join(
        cache_dir,
        "embeddings/",
        f"{dataset_name}_embeddings_{category_name}_{embedding_method}{'_combinations' if embedding_type == 'Combination' else ''}_seed_{seed}_sample_{sample_type}.pkl",
    )
    labels_file = os.path.join(
        cache_dir,
        "embeddings/",
        f"{dataset_name}_labels_{category_name}_{embedding_method}{'_combinations' if embedding_type == 'Combination' else ''}_seed_{seed}_sample_{sample_type}.pkl",
    )
    sim_file = os.path.join(
        cache_dir,
        "embeddings/",
        f"{dataset_name}_cosine_similarities_{category_name}_{embedding_method}{'_combinations' if embedding_type == 'Combination' else ''}_seed_{seed}_sample_{sample_type}.pkl",
    )

    if not os.path.exists(embeddings_file) or not os.path.exists(labels_file) or not os.path.exists(sim_file):
        return {}

    with open(embeddings_file, "rb") as f:
        embeddings = pickle.load(f)
    with open(labels_file, "rb") as f:
        labels = pickle.load(f)
    with open(sim_file, "rb") as f:
        similarities = pickle.load(f)

    data_store = {
        "embeddings": embeddings.tolist(),
        "labels": labels,
        "sample_type": sample_type,
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
    sample_type = embedding_data_store["sample_type"]
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

    df_plot = pd.DataFrame(
        {"x": embeddings_2d[:, 0], "y": embeddings_2d[:, 1], "label": labels, "Sample Type": sample_type}
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

    indices = np.where(similarities >= cosine_threshold)
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
