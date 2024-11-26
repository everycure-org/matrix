import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
import openai
import torch
from transformers import AutoTokenizer, AutoModel
import pickle
import gc
import logging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
import asyncio
import aiohttp
from tqdm.auto import tqdm
import time
from sklearn.metrics.pairwise import cosine_similarity

MAX_RETRIES = 3
BATCH_SIZE = 100
NORM_URL = "https://nodenorm.transltr.io/1.5/get_normalized_nodes"
API_URL = "https://annotator.ci.transltr.io/curie/"

configure_project("matrix")

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


def get_text_representation(row):
    global missing_data_rows_dict
    fields = [row.get("all_names:string[]", ""), row.get("all_categories:string[]", "")]
    field_names = ["all_names:string[]", "all_categories:string[]"]

    missing_fields = [
        field_name
        for field_name, field_value in zip(field_names, fields)
        if pd.isnull(field_value) or not str(field_value).strip()
    ]

    for missing_field in missing_fields:
        if missing_field not in missing_data_rows_dict:
            missing_data_rows_dict[missing_field] = []
        missing_data_rows_dict[missing_field].append(row)

    text_fields = []
    for field_value in fields:
        parsed_list = parse_list_string(field_value)
        text_fields.extend(parsed_list)
    text_representation = " ".join(text_fields).strip()
    if not text_representation:
        logging.warning(f"Empty text representation for row with index {row.name}")
    return text_representation


def get_text_representations(row):
    global missing_data_rows_dict
    names_field = "all_names:string[]"
    categories_field = "all_categories:string[]"

    names_field_value = row.get(names_field, "")
    categories_field_value = row.get(categories_field, "")

    missing_fields = []
    if pd.isnull(names_field_value) or not str(names_field_value).strip():
        missing_fields.append(names_field)
    if pd.isnull(categories_field_value) or not str(categories_field_value).strip():
        missing_fields.append(categories_field)

    for missing_field in missing_fields:
        if missing_field not in missing_data_rows_dict:
            missing_data_rows_dict[missing_field] = []
        missing_data_rows_dict[missing_field].append(row)

    names_list = parse_list_string(names_field_value)
    categories_list = parse_list_string(categories_field_value)

    if not names_list or not categories_list:
        logging.warning(f"Empty names or categories list for row with index {row.name}")
        return []

    from itertools import product

    combinations = list(product(names_list, categories_list))

    text_representations = [" ".join(combination).strip() for combination in combinations]
    return text_representations


def get_openai_embedding(texts):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = openai.embeddings.create(input=texts, model="text-embedding-3-small")
            embeddings = [item["embedding"] for item in response.model_dump()["data"]]
            return np.array(embeddings)
        except openai.error.RateLimitError as e:
            retries += 1
            logging.error(f"Rate limit error: {e}. Retrying {retries}/{MAX_RETRIES}")
            time.sleep(2**retries)
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            retries += 1
            time.sleep(2**retries)
    logging.error("Max retries exceeded for get_openai_embedding")
    return None


def process_model_combinations(model_name, model_info, datasets, cache_dir, seed):
    embeddings_dict = {}
    labels_dict = {}
    similarities_dict = {}
    if model_info["type"] == "hf":
        model, tokenizer = load_model_and_tokenizer(model_info)
    for category_name, df in tqdm(
        datasets.items(), desc=f"Processing model {model_name} with combinations", total=len(datasets), leave=False
    ):
        if df.empty:
            logging.warning(
                f"The DataFrame for category '{category_name}' " f"is empty. Skipping processing for this category."
            )
            continue

        cache_file = os.path.join(cache_dir, f"embeddings_{category_name}_{model_name}_combinations_seed_{seed}.pkl")
        labels_file = os.path.join(cache_dir, f"labels_{category_name}_{model_name}_combinations_seed_{seed}.pkl")
        sim_file = os.path.join(
            cache_dir, f"cosine_similarities_{category_name}_{model_name}_combinations_seed_{seed}.pkl"
        )

        if os.path.exists(cache_file) and os.path.exists(labels_file) and os.path.exists(sim_file):
            with open(cache_file, "rb") as f:
                embeddings = pickle.load(f)
            with open(labels_file, "rb") as f:
                labels = pickle.load(f)
            with open(sim_file, "rb") as f:
                similarities = pickle.load(f)
            embeddings_dict[category_name] = embeddings
            labels_dict[category_name] = labels
            similarities_dict[category_name] = similarities
            continue

        all_texts = []
        mappings = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing texts", leave=False):
            text_representations = get_text_representations(row)
            for combo_idx, text in enumerate(text_representations):
                all_texts.append(text)
                mappings.append({"row_idx": idx, "combination_idx": combo_idx})

        if not all_texts:
            logging.warning(f"No texts to process for category '{category_name}'. Skipping.")
            continue

        if model_info["type"] == "hf":
            embeddings = compute_embeddings_hf(model, tokenizer, all_texts)
            embeddings_dict[category_name] = embeddings
        elif model_info["type"] == "openai":
            batch_size = 500
            embeddings = []
            num_batches = (len(all_texts) + batch_size - 1) // batch_size
            valid_labels = []
            for i in tqdm(
                range(0, len(all_texts), batch_size),
                desc=f"Computing OpenAI embeddings for {category_name}",
                total=num_batches,
                leave=False,
            ):
                batch_texts = all_texts[i : i + batch_size]
                batch_mappings = mappings[i : i + batch_size]
                try:
                    batch_embeddings = get_openai_embedding(batch_texts)
                    if batch_embeddings is not None:
                        embeddings.append(batch_embeddings)
                        batch_labels = [
                            f"Row {mapping['row_idx']}, Combination {mapping['combination_idx']}: {text}"
                            for text, mapping in zip(batch_texts, batch_mappings)
                        ]
                        valid_labels.extend(batch_labels)
                except Exception as e:
                    logging.error(f"Failed to get embeddings for batch {i}: {e}")
                    continue

            if embeddings:
                embeddings = np.vstack(embeddings)
                embeddings_dict[category_name] = embeddings
                labels_dict[category_name] = valid_labels
            else:
                logging.error(f"No embeddings were generated for {category_name} with model {model_name}.")
                continue

        labels = [
            f"Row {mapping['row_idx']}, Combination {mapping['combination_idx']}: {text}"
            for text, mapping in zip(all_texts, mappings)
        ]
        labels_dict[category_name] = labels

        similarities = cosine_similarity(embeddings)
        similarities_dict[category_name] = similarities

    if model_info["type"] == "hf":
        unload_model_and_tokenizer(model, tokenizer)

    for category_name in embeddings_dict:
        cache_file = os.path.join(cache_dir, f"embeddings_{category_name}_{model_name}_combinations_seed_{seed}.pkl")
        labels_file = os.path.join(cache_dir, f"labels_{category_name}_{model_name}_combinations_seed_{seed}.pkl")
        sim_file = os.path.join(
            cache_dir, f"cosine_similarities_{category_name}_{model_name}_combinations_seed_{seed}.pkl"
        )
        with open(cache_file, "wb") as f:
            pickle.dump(embeddings_dict[category_name], f)
        with open(labels_file, "wb") as f:
            pickle.dump(labels_dict[category_name], f)
        with open(sim_file, "wb") as f:
            pickle.dump(similarities_dict[category_name], f)

    return model_name, embeddings_dict


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
    # 'PubMedBERT': {
    #     'type': 'hf',
    #     'tokenizer_name': "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    #     'model_name': "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    # },
    # 'SapBERT': {
    #     'type': 'hf',
    #     'tokenizer_name': "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    #     'model_name': "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    # },
    # 'BlueBERT': {
    #     'type': 'hf',
    #     'tokenizer_name': "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
    #     'model_name': "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
    # },
    # 'BioBERT': {
    #     'type': 'hf',
    #     'tokenizer_name': "dmis-lab/biobert-base-cased-v1.1",
    #     'model_name': "dmis-lab/biobert-base-cased-v1.1"
    # },
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


def process_model(model_name, model_info, datasets, cache_dir, seed):
    embeddings_dict = {}
    labels_dict = {}
    similarities_dict = {}
    if model_info["type"] == "hf":
        model, tokenizer = load_model_and_tokenizer(model_info)
    for category_name, df in tqdm(
        datasets.items(), desc=f"Processing model {model_name}", total=len(datasets), leave=False
    ):
        if df.empty:
            logging.warning(
                f"The DataFrame for category '{category_name}' " f"is empty. Skipping processing for this category."
            )
            continue
        cache_file = os.path.join(cache_dir, f"embeddings_{category_name}_{model_name}_seed_{seed}.pkl")
        labels_file = os.path.join(cache_dir, f"labels_{category_name}_{model_name}_seed_{seed}.pkl")
        sim_file = os.path.join(cache_dir, f"cosine_similarities_{category_name}_{model_name}_seed_{seed}.pkl")

        if os.path.exists(cache_file) and os.path.exists(labels_file) and os.path.exists(sim_file):
            with open(cache_file, "rb") as f:
                embeddings = pickle.load(f)
            with open(labels_file, "rb") as f:
                labels = pickle.load(f)
            with open(sim_file, "rb") as f:
                similarities = pickle.load(f)
            embeddings_dict[category_name] = embeddings
            labels_dict[category_name] = labels
            similarities_dict[category_name] = similarities
            continue

        texts = df.apply(get_text_representation, axis=1).tolist()
        labels = df.apply(lambda row: "<br>".join(f"{k}: {str(v)[:200]}" for k, v in row.items()), axis=1).tolist()
        if not texts:
            logging.warning(f"No texts to process for category '{category_name}'. Skipping.")
            continue
        if model_info["type"] == "hf":
            embeddings = compute_embeddings_hf(model, tokenizer, texts)
            embeddings_dict[category_name] = embeddings
        elif model_info["type"] == "openai":
            batch_size = 500
            embeddings = []
            num_batches = (len(texts) + batch_size - 1) // batch_size
            valid_labels = []
            for i in tqdm(
                range(0, len(texts), batch_size),
                desc=f"Computing OpenAI embeddings for {category_name}",
                total=num_batches,
                leave=False,
            ):
                batch_texts = texts[i : i + batch_size]
                batch_labels = labels[i : i + batch_size]
                try:
                    batch_embeddings = get_openai_embedding(batch_texts)
                    if batch_embeddings is not None:
                        embeddings.append(batch_embeddings)
                        valid_labels.extend(batch_labels)
                except Exception as e:
                    logging.error(f"Failed to get embeddings for batch {i}: {e}")
                    continue

            if embeddings:
                embeddings = np.vstack(embeddings)
                embeddings_dict[category_name] = embeddings
                labels_dict[category_name] = valid_labels
            else:
                logging.error(f"No embeddings were generated for {category_name} with model {model_name}.")
                continue

        labels_dict[category_name] = labels

        # Compute cosine similarities
        similarities = cosine_similarity(embeddings)
        similarities_dict[category_name] = similarities

    if model_info["type"] == "hf":
        unload_model_and_tokenizer(model, tokenizer)

    for category_name in embeddings_dict:
        cache_file = os.path.join(cache_dir, f"embeddings_{category_name}_{model_name}_seed_{seed}.pkl")
        labels_file = os.path.join(cache_dir, f"labels_{category_name}_{model_name}_seed_{seed}.pkl")
        sim_file = os.path.join(cache_dir, f"cosine_similarities_{category_name}_{model_name}_seed_{seed}.pkl")
        with open(cache_file, "wb") as f:
            pickle.dump(embeddings_dict[category_name], f)
        with open(labels_file, "wb") as f:
            pickle.dump(labels_dict[category_name], f)
        with open(sim_file, "wb") as f:
            pickle.dump(similarities_dict[category_name], f)

    return model_name, embeddings_dict


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
        except Exception:
            retries += 1
            await asyncio.sleep(2**retries)
    return None


async def batch_normalize_curies_async(category_curies, normalized_data, failed_ids):
    conn = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = []
        total_batches = 0
        for curies in category_curies.values():
            total_batches += (len(curies) + BATCH_SIZE - 1) // BATCH_SIZE
        with tqdm(total=total_batches, desc="Normalizing nodes", leave=False) as pbar:
            for curies in category_curies.values():
                for i in range(0, len(curies), BATCH_SIZE):
                    batch = curies[i : i + BATCH_SIZE]
                    payload = {"curies": batch, "conflate": False, "expand_all": True}
                    task = asyncio.ensure_future(normalize_node(session, NORM_URL, payload))
                    tasks.append(task)
            responses = []
            for f in tqdm(
                asyncio.as_completed(tasks), total=len(tasks), desc="Processing normalization responses", leave=False
            ):
                response = await f
                responses.append(response)
                pbar.update(1)
            for response in responses:
                if response:
                    for key, value in response.items():
                        if isinstance(value, dict):
                            equivalents = value.get("equivalent_identifiers", [])
                            normalized_equivalents = []
                            for eq in equivalents:
                                identifier = eq.get("identifier")
                                label = eq.get("label", "")
                                types = value.get("type", [])
                                types = list(map(lambda x: x.strip('"'), types))
                                if identifier:
                                    normalized_equivalents.append(
                                        {"identifier": identifier, "label": label, "types": types}
                                    )
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
        equivalent_items = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing category {category}", leave=False):
            id_column = "id:ID"
            equivalent_curies_column = "equivalent_curies:string[]"

            ids_to_normalize = [row[id_column]]

            equivalent_curies = row.get(equivalent_curies_column, [])
            equivalent_curies_list = parse_list_string(equivalent_curies)
            ids_to_normalize.extend(equivalent_curies_list)
            ids_to_normalize = list(set(ids_to_normalize))

            normalized_equivalents = []
            for id_value in ids_to_normalize:
                if id_value in normalized_data:
                    equivalents = normalized_data[id_value]
                    for eq in equivalents:
                        normalized_equivalents.append(
                            {"identifier": eq["identifier"], "label": eq["label"], "category": eq.get("types", [])}
                        )
                else:
                    normalized_equivalents.append(
                        {
                            "identifier": id_value,
                            "label": row.get("name", ""),
                            "category": parse_list_string(row.get("category", "")),
                        }
                    )

            unique_labels = {}
            for eq in normalized_equivalents:
                label = eq["label"]
                if label not in unique_labels:
                    unique_labels[label] = eq

            for eq in unique_labels.values():
                equivalent_item = row.copy()
                equivalent_item["id:ID"] = eq["identifier"]
                equivalent_item["name"] = eq["label"]
                equivalent_item["category"] = "ǂ".join(eq["category"]) if eq.get("category") else ""
                equivalent_items.append(equivalent_item)
        if equivalent_items:
            equivalent_df = pd.DataFrame(equivalent_items)
            equivalent_dfs[category] = equivalent_df
    return equivalent_dfs


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    project_path = str(Path.cwd())
    openai.api_key = os.getenv("OPENAI_API_KEY")
    seed1 = 54321
    seed2 = 67890
    cache_dir = "cached_datasets"
    os.makedirs(cache_dir, exist_ok=True)
    categories_file = os.path.join(cache_dir, "categories.pkl")

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

    datasets = {}
    positive_datasets = {}

    need_to_load_df = False
    for category in categories:
        positive_csv_filename = os.path.join(cache_dir, f"sampled_df_positives_{category}_seed_{seed1}.csv")
        negative_csv_filename = os.path.join(cache_dir, f"sampled_df_{category}_seed_{seed2}.csv")
        if not (os.path.exists(positive_csv_filename) and os.path.exists(negative_csv_filename)):
            need_to_load_df = True
            break

    if need_to_load_df:
        with KedroSession.create(project_path=project_path) as session:
            context = session.load_context()
            catalog = context.catalog
            df = catalog.load("ingestion.raw.rtx_kg2.nodes@pandas")

    for category in tqdm(categories, desc="Processing categories", leave=False):
        positive_csv_filename = os.path.join(cache_dir, f"sampled_df_positives_{category}_seed_{seed1}.csv")
        negative_csv_filename = os.path.join(cache_dir, f"sampled_df_{category}_seed_{seed2}.csv")

        if os.path.exists(positive_csv_filename) and os.path.exists(negative_csv_filename):
            positive_df = pd.read_csv(positive_csv_filename)
            negative_df = pd.read_csv(negative_csv_filename)
        else:
            if category == "All Categories":
                category_df = df.copy()
            else:
                category_df = df[df["category"] == category].copy()

            positive_n = min(30, len(category_df))
            positive_df = category_df.sample(n=positive_n, random_state=seed1)
            remaining_df = category_df.drop(positive_df.index)

            negative_n = min(70, len(remaining_df))
            negative_df = remaining_df.sample(n=negative_n, random_state=seed2)

            positive_df.to_csv(positive_csv_filename, index=False)
            negative_df.to_csv(negative_csv_filename, index=False)

        datasets[category] = negative_df.reset_index(drop=True)
        positive_datasets[category] = positive_df.reset_index(drop=True)

    category_curies = {}
    for category, df in positive_datasets.items():
        id_column = "id:ID"
        equivalent_curies_column = "equivalent_curies:string[]"
        if id_column not in df.columns:
            continue
        curies_set = set()
        id_values = df[id_column].dropna().astype(str).tolist()
        equivalent_curies_values = df[equivalent_curies_column].dropna().tolist()
        for eq_curies in equivalent_curies_values:
            eq_curies_list = parse_list_string(eq_curies)
            curies_set.update(eq_curies_list)
        curies_set.update(id_values)
        category_curies[category] = list(curies_set)

    normalized_data = {}
    failed_ids = set()
    asyncio.run(batch_normalize_curies_async(category_curies, normalized_data, failed_ids))

    with open(os.path.join(cache_dir, f"normalized_nodes_seed_{seed1}.pkl"), "wb") as f:
        pickle.dump(normalized_data, f)

    equivalent_dfs = create_equivalent_items_dfs(positive_datasets, normalized_data)

    for category, df in equivalent_dfs.items():
        output_csv_filename = os.path.join(cache_dir, f"equivalent_items_{category}_seed_{seed1}.csv")
        df.to_csv(output_csv_filename, index=False)
        positive_datasets[category] = df  # Update positive_datasets with normalized data

    # Process models without combinations
    for model_name, model_info in tqdm(embedding_models_info.items(), desc="Processing models", leave=False):
        model_name, embeddings_dict = process_model(model_name, model_info, datasets, cache_dir, seed2)

    # Process models with combinations
    for model_name, model_info in tqdm(
        embedding_models_info.items(), desc="Processing models with combinations", leave=False
    ):
        model_name, embeddings_dict = process_model_combinations(model_name, model_info, datasets, cache_dir, seed2)

    # Similarly for positive datasets
    for model_name, model_info in tqdm(
        embedding_models_info.items(), desc="Processing models for positive datasets", leave=False
    ):
        model_name, embeddings_dict = process_model(model_name, model_info, positive_datasets, cache_dir, seed1)

    for model_name, model_info in tqdm(
        embedding_models_info.items(), desc="Processing models for positive datasets with combinations", leave=False
    ):
        model_name, embeddings_dict = process_model_combinations(
            model_name, model_info, positive_datasets, cache_dir, seed1
        )

    del datasets
    del positive_datasets
    del equivalent_dfs

    if missing_data_rows_dict:
        for missing_field, rows in missing_data_rows_dict.items():
            missing_data_df = pd.DataFrame(rows)
            missing_data_csv = os.path.join(cache_dir, f"missing_data_{missing_field}.csv")
            missing_data_df.to_csv(missing_data_csv, index=False)


if __name__ == "__main__":
    main()
