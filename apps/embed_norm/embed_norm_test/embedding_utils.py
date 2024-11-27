import os
import json
import logging
import gc
import time
import pickle
import numpy as np
import pandas as pd
import openai
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
import aiohttp
from tqdm.auto import tqdm
from kedro.framework.session import KedroSession
import curategpt

MAX_RETRIES = 3
BATCH_SIZE = 100
NORM_URL = "https://nodenorm.transltr.io/1.5/get_normalized_nodes"
API_URL = "https://annotator.ci.transltr.io/curie/"
missing_data_rows_dict = {}


def parse_list_string(s):
    if isinstance(s, list):
        return s
    elif isinstance(s, str):
        s = s.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                pass
        if "ǂ" in s:
            return s.split("ǂ")
        return [s]
    else:
        return []


def get_text_representation(row, text_fields=None, combine_fields=False):
    if text_fields is None:
        text_fields = ["all_names:string[]", "all_categories:string[]"]
    global missing_data_rows_dict
    fields = [row.get(field, "") for field in text_fields]
    missing_fields = [field for field, value in zip(text_fields, fields) if pd.isnull(value) or not str(value).strip()]
    for missing_field in missing_fields:
        if missing_field not in missing_data_rows_dict:
            missing_data_rows_dict[missing_field] = []
        missing_data_rows_dict[missing_field].append(row)
    if combine_fields:
        parsed_lists = [parse_list_string(field_value) for field_value in fields]
        from itertools import product

        combinations = list(product(*parsed_lists))
        text_representations = [" ".join(combination).strip() for combination in combinations]
        return text_representations
    else:
        text_values = []
        for field_value in fields:
            parsed_list = parse_list_string(field_value)
            text_values.extend(parsed_list)
        text_representation = " ".join(text_values).strip()
        if not text_representation:
            logging.warning(f"Empty text representation for row with index {row.name}")
        return text_representation


def get_openai_embedding(texts):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = openai.embeddings.create(input=texts, model="text-embedding-3-large")
            embeddings = [item["embedding"] for item in response.model_dump()["data"]]
            return np.array(embeddings)
        except openai.RateLimitError as e:
            retries += 1
            logging.error(f"Rate limit error: {e}. Retrying {retries}/{MAX_RETRIES}")
            time.sleep(2**retries)
        except openai.APIStatusError as e:
            logging.error(f"Invalid request error: {e}")
            # Return None since retrying won't help
            return None
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            retries += 1
            time.sleep(2**retries)
    logging.error("Max retries exceeded for get_openai_embedding")
    return None


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


def compute_embeddings_hf(model, tokenizer, texts, batch_size=16, max_length=256):
    if not texts:
        logging.warning("No texts provided for embedding computation.")
        return np.array([])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = []
    dataset = list(zip(texts))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    for batch in tqdm(dataloader, desc="Computing embeddings", leave=False):
        batch_texts = batch[0]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.append(batch_embeddings)
    if embeddings:
        embeddings = np.vstack(embeddings)
    else:
        logging.warning("No embeddings were computed.")
        embeddings = np.array([])
    return embeddings


def cache_data(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_cached_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data
    else:
        return None


def process_model(
    model_name,
    model_info,
    datasets,
    cache_dir,
    seed,
    text_fields=None,
    text_representation_func=None,
    label_generation_func=None,
    dataset_name="default",
    use_ontogpt=False,
    cache_suffix="",
    use_combinations=False,
    combine_fields=False,
):
    embeddings_dict = {}
    labels_dict = {}
    similarities_dict = {}

    if model_info["type"] == "hf":
        model, tokenizer = load_model_and_tokenizer(model_info)

    for category_name, df in tqdm(
        datasets.items(), desc=f"Processing model {model_name}", total=len(datasets), leave=False
    ):
        if df.empty:
            logging.warning(f"The DataFrame for category '{category_name}' is empty. Skipping.")
            continue

        # Prepare texts using vectorized operations

        # TODO: handle text representation function in notebook
        # if text_representation_func is None:
        #     texts = df.apply(get_text_representation, axis=1).tolist()
        # else:
        #     texts = df.apply(text_representation_func, axis=1).tolist()
        if use_ontogpt:
            base_texts = df.apply(lambda row: get_text_representation(row, text_fields, False), axis=1)
            enhanced_info = base_texts.apply(curategpt.extract)
            all_texts = (base_texts + " " + enhanced_info).str.strip()
        elif use_combinations:
            # Handle combinations
            # This may require additional handling to vectorize
            pass
        else:
            all_texts = df.apply(lambda row: get_text_representation(row, text_fields, False), axis=1)

        if all_texts.empty:
            logging.warning(f"No texts to process for category '{category_name}'. Skipping.")
            continue

        # Compute embeddings
        if model_info["type"] == "hf":
            embeddings = compute_embeddings_hf(model, tokenizer, all_texts.tolist())
            embeddings_dict[category_name] = embeddings
        elif model_info["type"] == "openai":
            embeddings = get_openai_embeddings_in_batches(all_texts.tolist())
            embeddings_dict[category_name] = embeddings

        # Generate labels
        if label_generation_func is None:
            labels = [f"Row {idx}: {text}" for idx, text in enumerate(all_texts)]
        else:
            labels = df.apply(label_generation_func, axis=1).tolist()
        labels_dict[category_name] = labels

        # Compute similarities
        similarities = cosine_similarity(embeddings)
        similarities_dict[category_name] = similarities

        # Cache results
        cache_results(
            category_name,
            model_name,
            embeddings_dict,
            labels_dict,
            similarities_dict,
            cache_dir,
            dataset_name,
            seed,
            cache_suffix,
            use_combinations,
        )

    if model_info["type"] == "hf":
        unload_model_and_tokenizer(model, tokenizer)

    return model_name, embeddings_dict


def cache_results(
    category_name,
    model_name,
    embeddings_dict,
    labels_dict,
    similarities_dict,
    cache_dir,
    dataset_name,
    seed,
    cache_suffix,
    use_combinations,
):
    combinations_suffix = "_combinations" if use_combinations else ""
    # Ensure the cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Construct file paths
    cache_file = os.path.join(
        cache_dir,
        f"{dataset_name}_embeddings_{category_name}_{model_name}{combinations_suffix}_seed_{seed}{cache_suffix}.pkl",
    )
    labels_file = os.path.join(
        cache_dir,
        f"{dataset_name}_labels_{category_name}_{model_name}{combinations_suffix}_seed_{seed}{cache_suffix}.pkl",
    )
    sim_file = os.path.join(
        cache_dir,
        f"{dataset_name}_cosine_similarities_{category_name}_{model_name}{combinations_suffix}_seed_{seed}{cache_suffix}.pkl",
    )

    # Cache embeddings
    with open(cache_file, "wb") as f:
        pickle.dump(embeddings_dict[category_name], f)
    # Cache labels
    with open(labels_file, "wb") as f:
        pickle.dump(labels_dict[category_name], f)
    # Cache similarities
    with open(sim_file, "wb") as f:
        pickle.dump(similarities_dict[category_name], f)


# Helper function to get OpenAI embeddings in batches
def get_openai_embeddings_in_batches(texts, batch_size=20):  # Adjusted batch size
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing OpenAI embeddings", leave=False):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = get_openai_embedding(batch_texts)
        if batch_embeddings is not None:
            embeddings.append(batch_embeddings)
        else:
            logging.warning(f"Skipping batch {i}-{i+batch_size} due to errors")
    if embeddings:
        return np.vstack(embeddings)
    else:
        logging.error("No embeddings were computed.")
        return np.array([])


def process_models(
    model_names,
    datasets,
    cache_dir,
    seed,
    text_fields=None,
    label_generation_func=None,
    dataset_name="default",
    use_ontogpt=False,
    cache_suffix="",
    use_combinations=False,
    combine_fields=False,
):
    embeddings_dict_all_models = {}
    # labels_dict_all_models = {}
    # similarities_dict_all_models = {}
    # Process models to minimize loading/unloading
    # First, process OpenAI models (no loading required)
    openai_models = [name for name in model_names if embedding_models_info[name]["type"] == "openai"]
    for model_name in openai_models:
        model_info = embedding_models_info[model_name]
        _, embeddings_dict = process_model(
            model_name=model_name,
            model_info=model_info,
            datasets=datasets,
            cache_dir=cache_dir,
            seed=seed,
            text_fields=text_fields,
            label_generation_func=label_generation_func,
            dataset_name=dataset_name,
            use_ontogpt=use_ontogpt,
            cache_suffix=cache_suffix,
            use_combinations=use_combinations,
            combine_fields=combine_fields,
        )
        embeddings_dict_all_models[model_name] = embeddings_dict
    # Then process HF models, loading and unloading each model
    hf_models = [name for name in model_names if embedding_models_info[name]["type"] == "hf"]
    for model_name in hf_models:
        model_info = embedding_models_info[model_name]
        _, embeddings_dict = process_model(
            model_name=model_name,
            model_info=model_info,
            datasets=datasets,
            cache_dir=cache_dir,
            seed=seed,
            text_fields=text_fields,
            label_generation_func=label_generation_func,
            dataset_name=dataset_name,
            use_ontogpt=use_ontogpt,
            cache_suffix=cache_suffix,
            use_combinations=use_combinations,
            combine_fields=combine_fields,
        )
        embeddings_dict_all_models[model_name] = embeddings_dict
    return embeddings_dict_all_models


def generate_candidate_pairs(similarities, threshold=0.8):
    candidate_pairs = []
    num_nodes = similarities.shape[0]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if similarities[i, j] >= threshold:
                candidate_pairs.append((i, j))
    return candidate_pairs


def refine_candidate_mappings_with_llm(candidate_pairs, nodes_df):
    refined_mappings = []
    for i, j in candidate_pairs:
        node_i = nodes_df.iloc[i]
        node_j = nodes_df.iloc[j]
        prompt = f"Do the following two entities represent the same concept?\n\nEntity 1:\n{node_i.to_json()}\n\nEntity 2:\n{node_j.to_json()}\n\nAnswer 'Yes' or 'No'."
        response = openai.Completion.create(engine="gpt-3.5-turbo", prompt=prompt, max_tokens=5, temperature=0)
        answer = response.choices[0].text.strip().lower()
        if answer == "yes":
            refined_mappings.append((i, j))
    return refined_mappings


def find_additional_mappings_with_curategpt(nodes_df):
    additional_mappings = []
    for idx, row in nodes_df.iterrows():
        matches = curategpt.match_agent(row.to_dict())
        for match in matches:
            additional_mappings.append((idx, match["index"]))
    return additional_mappings


async def normalize_node(session, url, payload):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    retries += 1
                    await asyncio.sleep(2**retries)
        except aiohttp.ClientError:
            retries += 1
            await asyncio.sleep(2**retries)
    return None


async def batch_normalize_curies_async(category_curies, normalized_data, failed_ids):
    # Flatten all curies into a single list
    all_curies = [curie for curies in category_curies.values() for curie in curies]
    total_batches = (len(all_curies) + BATCH_SIZE - 1) // BATCH_SIZE

    conn = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = []
        with tqdm(total=total_batches, desc="Normalizing nodes", leave=False) as pbar:
            for i in range(0, len(all_curies), BATCH_SIZE):
                batch = all_curies[i : i + BATCH_SIZE]
                payload = {"curies": batch, "conflate": False, "expand_all": True}
                task = asyncio.ensure_future(normalize_node(session, NORM_URL, payload))
                tasks.append(task)
                pbar.update(1)

        responses = await asyncio.gather(*tasks)

    # Process all responses
    for response in tqdm(responses, desc="Processing normalization responses", leave=False):
        if response:
            for key, value in response.items():
                if isinstance(value, dict):
                    equivalents = value.get("equivalent_identifiers", [])
                    types = [t.strip('"') for t in value.get("type", [])]
                    normalized_equivalents = [
                        {"identifier": eq.get("identifier"), "label": eq.get("label", ""), "types": types}
                        for eq in equivalents
                        if eq.get("identifier")
                    ]
                    if normalized_equivalents:
                        normalized_data[key] = normalized_equivalents
                    else:
                        failed_ids.add(key)
                else:
                    failed_ids.add(key)


def create_equivalent_items_dfs(positive_datasets, normalized_data):
    equivalent_dfs = {}
    for category, df in tqdm(
        positive_datasets.items(),
        total=len(positive_datasets),
        desc="Creating equivalent items DataFrames",
        leave=False,
    ):
        # Prepare ids_to_normalize per row
        df["ids_to_normalize"] = df.apply(
            lambda row: list(set([row["id:ID"]] + parse_list_string(row.get("equivalent_curies:string[]", [])))), axis=1
        )

        # Explode the DataFrame to have one id per row
        df_exploded = df.explode("ids_to_normalize").reset_index(drop=True)
        df_exploded.rename(columns={"ids_to_normalize": "id_to_normalize"}, inplace=True)

        # Create a DataFrame for all unique ids_to_normalize
        unique_ids = df_exploded["id_to_normalize"].unique()
        equivalents_list = []

        # Map ids to their normalized equivalents
        for id_value in unique_ids:
            if id_value in normalized_data:
                equivalents = normalized_data[id_value]
                for eq in equivalents:
                    equivalents_list.append(
                        {
                            "id_to_normalize": id_value,
                            "identifier": eq["identifier"],
                            "label": eq["label"],
                            "category_list": eq.get("types", []),
                        }
                    )
            else:
                # Get corresponding rows in df_exploded
                rows_with_id = df_exploded[df_exploded["id_to_normalize"] == id_value]
                for _, row in rows_with_id.iterrows():
                    equivalents_list.append(
                        {
                            "id_to_normalize": id_value,
                            "identifier": id_value,
                            "label": row.get("name", ""),
                            "category_list": parse_list_string(row.get("category", "")),
                        }
                    )

        equivalents_df = pd.DataFrame(equivalents_list)
        equivalents_df = equivalents_df.drop_duplicates(subset=["label"])

        # Merge back with df_exploded
        merged_df = df_exploded.merge(equivalents_df, on="id_to_normalize", how="left")
        merged_df["id:ID"] = merged_df["identifier"]
        merged_df["name"] = merged_df["label"]
        merged_df["category"] = merged_df["category_list"].apply(lambda x: "ǂ".join(x) if x else "")
        equivalent_df = merged_df[df.columns].drop_duplicates()

        equivalent_dfs[category] = equivalent_df
    return equivalent_dfs


def load_categories(cache_dir, dataset_name, nodes_dataset_name):
    categories_file = os.path.join(cache_dir, f"{dataset_name}_categories.pkl")
    categories = load_cached_data(categories_file)
    if categories is None:
        with KedroSession.create() as session:
            context = session.load_context()
            catalog = context.catalog
            df = catalog.load(nodes_dataset_name)
        unique_categories = df["category"].unique().tolist()
        categories = unique_categories + ["All Categories"]
        cache_data(categories, categories_file)
    return categories


def load_datasets(
    cache_dir,
    dataset_name,
    nodes_dataset_name,
    edges_dataset_name,
    categories,
    seed1,
    seed2,
    total_sample_size=1000,
    positive_ratio=0.3,
):
    datasets = {}
    positive_datasets = {}
    need_to_load_df = False
    positive_n = round(total_sample_size * positive_ratio)
    negative_n = total_sample_size - positive_n
    cache_suffix = f"_pos_{positive_n}_neg_{negative_n}"
    for category in categories:
        positive_csv_filename = os.path.join(
            cache_dir, f"{dataset_name}_sampled_df_positives_{category}_seed_{seed1}{cache_suffix}.csv"
        )
        negative_csv_filename = os.path.join(
            cache_dir, f"{dataset_name}_sampled_df_{category}_seed_{seed2}{cache_suffix}.csv"
        )
        if not (os.path.exists(positive_csv_filename) and os.path.exists(negative_csv_filename)):
            need_to_load_df = True
            break
    if need_to_load_df:
        with KedroSession.create() as session:
            context = session.load_context()
            catalog = context.catalog
            nodes_df = catalog.load(nodes_dataset_name)
    else:
        nodes_df = None
    for category in tqdm(categories, desc="Processing categories", leave=False):
        positive_csv_filename = os.path.join(
            cache_dir, f"{dataset_name}_sampled_df_positives_{category}_seed_{seed1}{cache_suffix}.csv"
        )
        negative_csv_filename = os.path.join(
            cache_dir, f"{dataset_name}_sampled_df_{category}_seed_{seed2}{cache_suffix}.csv"
        )
        if os.path.exists(positive_csv_filename) and os.path.exists(negative_csv_filename):
            positive_df = pd.read_csv(positive_csv_filename)
            negative_df = pd.read_csv(negative_csv_filename)
        else:
            if nodes_df is None:
                with KedroSession.create() as session:
                    context = session.load_context()
                    catalog = context.catalog
                    nodes_df = catalog.load(nodes_dataset_name)
            if category == "All Categories":
                category_df = nodes_df.copy()
            else:
                category_df = nodes_df[nodes_df["category"] == category].copy()
            positive_n_actual = min(positive_n, len(category_df))
            positive_df = category_df.sample(n=positive_n_actual, random_state=seed1)
            remaining_df = category_df.drop(positive_df.index)
            negative_n_actual = min(negative_n, len(remaining_df))
            negative_df = remaining_df.sample(n=negative_n_actual, random_state=seed2)
            positive_df.to_csv(positive_csv_filename, index=False)
            negative_df.to_csv(negative_csv_filename, index=False)
        datasets[category] = negative_df.reset_index(drop=True)
        positive_datasets[category] = positive_df.reset_index(drop=True)
    return positive_datasets, datasets


def load_embeddings_and_labels(
    cache_dir, dataset_name, model_name, categories, seed, use_combinations=False, cache_suffix=""
):
    embeddings_dict = {}
    labels_dict = {}
    suffix = "_combinations" if use_combinations else ""
    for category_name in categories:
        cache_file = os.path.join(
            cache_dir, f"{dataset_name}_embeddings_{category_name}_{model_name}{suffix}_seed_{seed}{cache_suffix}.pkl"
        )
        labels_file = os.path.join(
            cache_dir, f"{dataset_name}_labels_{category_name}_{model_name}{suffix}_seed_{seed}{cache_suffix}.pkl"
        )
        embeddings = load_cached_data(cache_file)
        labels = load_cached_data(labels_file)
        if embeddings is not None and labels is not None:
            embeddings_dict[category_name] = embeddings
            labels_dict[category_name] = labels
        else:
            logging.warning(f"Embeddings or labels not found for category '{category_name}' with model '{model_name}'.")
    return embeddings_dict, labels_dict
