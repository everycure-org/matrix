import os
from pathlib import Path
import pandas as pd
import numpy as np
import openai
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from sklearn.cluster import KMeans
import pickle
from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
import gc
import logging

configure_project("matrix")


def get_openai_embedding(texts):
    try:
        response = openai.embeddings.create(input=texts, model="text-embedding-3-large")
        embeddings = [item["embedding"] for item in response.model_dump()["data"]]
        return np.array(embeddings)
    except Exception as e:
        logging.error(f"Error getting OpenAI embeddings: {e}")
        return None


def get_text_representation(row):
    text_representation = f"{row['name']} {row.get('description', '')} {row.get('all_names:string[]', '')} {row.get('all_categories:string[]', '')}"
    return text_representation


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


embedding_models_info = {
    "OpenAI": {
        "type": "openai",
    },
    "PubMedBERT": {
        "type": "hf",
        "tokenizer_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    },
    "SapBERT": {
        "type": "hf",
        "tokenizer_name": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "model_name": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    },
    "BlueBERT": {
        "type": "hf",
        "tokenizer_name": "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
        "model_name": "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
    },
    "BioBERT": {
        "type": "hf",
        "tokenizer_name": "dmis-lab/biobert-base-cased-v1.1",
        "model_name": "dmis-lab/biobert-base-cased-v1.1",
    },
}

embedding_models = list(embedding_models_info.keys())


def load_model_and_tokenizer(model_info):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_info["tokenizer_name"])
    model = AutoModel.from_pretrained(model_info["model_name"])
    model.to(device)
    model.eval()
    return model, tokenizer


def unload_model_and_tokenizer(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_embeddings_hf(model, tokenizer, texts, initial_batch_size=16, max_length=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    embeddings = []
    idx = 0
    total_texts = len(texts)
    batch_size = initial_batch_size
    while idx < total_texts:
        success = False
        while not success and batch_size > 0:
            try:
                batch_texts = texts[idx : idx + batch_size]
                inputs = tokenizer(
                    batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
                embeddings.extend(batch_embeddings)
                idx += batch_size
                success = True
                batch_size = initial_batch_size
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logging.warning(f"Out of memory at index {idx}, reducing batch size")
                    batch_size = batch_size // 2
                    torch.cuda.empty_cache()
                else:
                    logging.error(f"Runtime error at index {idx}: {e}")
                    break
        if batch_size == 0:
            logging.error("Batch size reduced to zero, stopping embedding computation.")
            break
    return np.array(embeddings)


def process_model(model_name, model_info, datasets, cache_dir, seed):
    embeddings_dict = {}
    if model_info["type"] == "hf":
        model, tokenizer = load_model_and_tokenizer(model_info)
    for category_name, df in datasets.items():
        cache_file = os.path.join(cache_dir, f"embeddings_{category_name}_{model_name}_seed_{seed}.pkl")
        if os.path.exists(cache_file):
            print(f"Loading embeddings from cache for model: {model_name}, category: {category_name}")
            with open(cache_file, "rb") as f:
                embeddings = pickle.load(f)
            embeddings_dict[category_name] = embeddings
            continue
        print(f"Processing embeddings for model: {model_name}, category: {category_name}")
        texts = df.apply(get_text_representation, axis=1).tolist()
        if model_info["type"] == "hf":
            embeddings = compute_embeddings_hf(model, tokenizer, texts)
        elif model_info["type"] == "openai":
            batch_size = 500
            embeddings = []
            idx = 0
            total_texts = len(texts)
            while idx < total_texts:
                batch_texts = texts[idx : idx + batch_size]
                batch_embeddings = get_openai_embedding(batch_texts)
                if batch_embeddings is not None:
                    embeddings.extend(batch_embeddings)
                else:
                    logging.error(f"Failed to get embeddings for batch starting at index {idx}")
                idx += batch_size
            embeddings = np.array(embeddings)
        embeddings_dict[category_name] = embeddings
        with open(cache_file, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"Saved embeddings to cache for model: {model_name}, category: {category_name}")
    if model_info["type"] == "hf":
        unload_model_and_tokenizer(model, tokenizer)
    return model_name, embeddings_dict


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    project_path = str(Path.cwd())
    openai.api_key = os.getenv("OPENAI_API_KEY")
    seed = 12345
    cache_dir = "cached_datasets"
    os.makedirs(cache_dir, exist_ok=True)
    categories_file = os.path.join(cache_dir, f"categories_seed_{seed}.pkl")
    if os.path.exists(categories_file):
        with open(categories_file, "rb") as f:
            categories = pickle.load(f)
    else:
        with KedroSession.create(project_path=project_path) as session:
            context = session.load_context()
            catalog = context.catalog
            df = catalog.load("ingestion.raw.rtx_kg2.nodes@pandas")
        unique_categories = df["category"].unique().tolist()
        categories = unique_categories + ["All Categories"]
        with open(categories_file, "wb") as f:
            pickle.dump(categories, f)
    if "df" not in locals():
        with KedroSession.create(project_path=project_path) as session:
            context = session.load_context()
            catalog = context.catalog
            df = catalog.load("ingestion.raw.rtx_kg2.nodes@pandas")
    datasets = {}
    for category in categories:
        csv_filename = os.path.join(cache_dir, f"sampled_df_{category}_seed_{seed}.csv")
        if os.path.exists(csv_filename):
            sampled_df = pd.read_csv(csv_filename)
        else:
            if category == "All Categories":
                category_df = df
            else:
                category_df = df[df["category"] == category]
            total_rows = category_df.shape[0]
            sample_fraction = min(1, 100 / total_rows)
            sampled_df = category_df.sample(frac=sample_fraction, random_state=seed)
            sampled_df = sampled_df.head(100)
            sampled_df.to_csv(csv_filename, index=False)
        datasets[category] = sampled_df
    print("Processing models...")
    embeddings_all = {}
    for model_name, model_info in embedding_models_info.items():
        model_name, embeddings_dict = process_model(model_name, model_info, datasets, cache_dir, seed)
        embeddings_all[model_name] = embeddings_dict
    labels_by_category = {}
    for category, sampled_df in datasets.items():
        labels = sampled_df["name"].tolist()
        labels_by_category[category] = labels
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    embedding_options = [{"label": name, "value": name} for name in embedding_models]
    dataset_options = [{"label": name, "value": name} for name in embeddings_all[next(iter(embeddings_all))].keys()]
    app.layout = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H1("Interactive Embedding Visualization"),
                            html.P("Select embedding method and adjust parameters."),
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
                                id="dataset", options=dataset_options, value="All Categories", clearable=False
                            ),
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Embedding Method"),
                            dcc.Dropdown(
                                id="embedding-method", options=embedding_options, value="OpenAI", clearable=False
                            ),
                        ],
                        width=3,
                    ),
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
                        width=3,
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
                        width=3,
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
                        width=3,
                    ),
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
                        width=3,
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
                        width=3,
                    ),
                ],
                className="my-4",
            ),
            dbc.Row(
                [
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
                        width=3,
                    ),
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
                        width=3,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Loading(
                                id="loading-embedding-plot", type="default", children=dcc.Graph(id="embedding-plot")
                            )
                        ],
                        width=12,
                    )
                ]
            ),
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

    reduced_embeddings_cache = {}

    @app.callback(
        Output("embedding-plot", "figure"),
        Input("dataset", "value"),
        Input("embedding-method", "value"),
        Input("dim-reduction-method", "value"),
        Input("n-neighbors", "value"),
        Input("min-dist", "value"),
        Input("perplexity", "value"),
        Input("point-size", "value"),
        Input("enable-clustering", "value"),
        Input("num-clusters", "value"),
    )
    def update_plot(
        dataset_name,
        embedding_method,
        dim_reduction_method,
        n_neighbors,
        min_dist,
        perplexity,
        point_size,
        enable_clustering,
        num_clusters,
    ):
        cache_key = (embedding_method, dataset_name, dim_reduction_method, n_neighbors, min_dist, perplexity)
        if cache_key in reduced_embeddings_cache:
            embeddings_2d = reduced_embeddings_cache[cache_key]
        else:
            embeddings = embeddings_all[embedding_method][dataset_name]
            if dim_reduction_method == "umap":
                embeddings_2d = reduce_dimensions_umap(embeddings, n_neighbors=n_neighbors, min_dist=min_dist)
            elif dim_reduction_method == "pca":
                embeddings_2d = reduce_dimensions(embeddings, method="pca")
            elif dim_reduction_method == "tsne":
                embeddings_2d = reduce_dimensions(embeddings, method="tsne", perplexity=perplexity)
            else:
                embeddings_2d = embeddings
            reduced_embeddings_cache[cache_key] = embeddings_2d
        labels = labels_by_category[dataset_name]
        sampled_df = datasets[dataset_name]
        df_plot = pd.DataFrame(
            {
                "x": embeddings_2d[:, 0],
                "y": embeddings_2d[:, 1],
                "label": labels,
                "description": sampled_df["description"].iloc[: len(embeddings_2d)].fillna(""),
                "category": sampled_df["category"].iloc[: len(embeddings_2d)].fillna(""),
            }
        )
        if enable_clustering:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            df_plot["cluster"] = kmeans.fit_predict(embeddings_2d)
            color = "cluster"
        else:
            color = None
        fig = px.scatter(
            df_plot,
            x="x",
            y="y",
            hover_data=["label", "description", "category"],
            color=color,
            title=f"{embedding_method} Embeddings Visualization ({dim_reduction_method.upper()}) - {dataset_name}",
            labels={"x": "Dimension 1", "y": "Dimension 2"},
        )
        if color is not None:
            fig.update_traces(marker=dict(size=point_size))
        else:
            fig.update_traces(marker=dict(size=point_size, color="blue"))
        fig.update_layout(
            hovermode="closest", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=False), yaxis=dict(showgrid=False)
        )
        return fig

    app.run_server(debug=True, use_reloader=False, host="0.0.0.0", port=8050)
