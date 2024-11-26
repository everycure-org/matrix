# main.py

import os
import logging
import asyncio
import pickle
import pandas as pd
from pathlib import Path
from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
from tqdm.auto import tqdm
import openai

from embedding_utils import (
    parse_list_string,
    process_model,
    process_model_combinations,
    batch_normalize_curies_async,
    create_equivalent_items_dfs,
    embedding_models_info,
    missing_data_rows_dict,
)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    project_path = str(Path.cwd())
    openai.api_key = os.getenv("OPENAI_API_KEY")
    seed1 = 54321
    seed2 = 67890
    cache_dir = "cached_datasets"
    os.makedirs(cache_dir, exist_ok=True)

    dataset_name = "rtx_kg2"  # Replace with your dataset name
    nodes_dataset_name = "ingestion.raw.rtx_kg2.nodes@pandas"  # Replace with your nodes dataset name
    # edges_dataset_name = "ingestion.raw.rtx_kg2.edges@pandas"  # Replace with your edges dataset name

    categories_file = os.path.join(cache_dir, f"{dataset_name}_categories.pkl")

    configure_project("matrix")

    if os.path.exists(categories_file):
        with open(categories_file, "rb") as f:
            categories = pickle.load(f)
    else:
        with KedroSession.create(project_path=project_path) as session:
            context = session.load_context()
            catalog = context.catalog
            df = catalog.load(nodes_dataset_name)
        unique_categories = df["category"].unique().tolist()
        categories = unique_categories + ["All Categories"]
        with open(categories_file, "wb") as f:
            pickle.dump(categories, f)

    datasets = {}
    positive_datasets = {}

    need_to_load_df = False
    for category in categories:
        positive_csv_filename = os.path.join(
            cache_dir, f"{dataset_name}_sampled_df_positives_{category}_seed_{seed1}.csv"
        )
        negative_csv_filename = os.path.join(cache_dir, f"{dataset_name}_sampled_df_{category}_seed_{seed2}.csv")
        if not (os.path.exists(positive_csv_filename) and os.path.exists(negative_csv_filename)):
            need_to_load_df = True
            break

    if need_to_load_df:
        with KedroSession.create(project_path=project_path) as session:
            context = session.load_context()
            catalog = context.catalog
            df = catalog.load(nodes_dataset_name)

    for category in tqdm(categories, desc="Processing categories", leave=False):
        positive_csv_filename = os.path.join(
            cache_dir, f"{dataset_name}_sampled_df_positives_{category}_seed_{seed1}.csv"
        )
        negative_csv_filename = os.path.join(cache_dir, f"{dataset_name}_sampled_df_{category}_seed_{seed2}.csv")

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

    with KedroSession.create(project_path=project_path) as session:
        context = session.load_context()
        catalog = context.catalog
        # edges_df = catalog.load(edges_dataset_name)

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

    with open(os.path.join(cache_dir, f"{dataset_name}_normalized_nodes_seed_{seed1}.pkl"), "wb") as f:
        pickle.dump(normalized_data, f)

    equivalent_dfs = create_equivalent_items_dfs(positive_datasets, normalized_data)

    for category, df in equivalent_dfs.items():
        output_csv_filename = os.path.join(cache_dir, f"{dataset_name}_equivalent_items_{category}_seed_{seed1}.csv")
        df.to_csv(output_csv_filename, index=False)
        positive_datasets[category] = df  # Update positive_datasets with normalized data

    for model_name, model_info in tqdm(embedding_models_info.items(), desc="Processing models", leave=False):
        process_model(model_name, model_info, datasets, cache_dir, seed2, dataset_name=dataset_name)

    for model_name, model_info in tqdm(
        embedding_models_info.items(), desc="Processing models with combinations", leave=False
    ):
        process_model_combinations(model_name, model_info, datasets, cache_dir, seed2, dataset_name=dataset_name)

    for model_name, model_info in tqdm(
        embedding_models_info.items(), desc="Processing models for positive datasets", leave=False
    ):
        process_model(model_name, model_info, positive_datasets, cache_dir, seed1, dataset_name=dataset_name)

    for model_name, model_info in tqdm(
        embedding_models_info.items(), desc="Processing models for positive datasets with combinations", leave=False
    ):
        process_model_combinations(
            model_name, model_info, positive_datasets, cache_dir, seed1, dataset_name=dataset_name
        )

    del datasets
    del positive_datasets
    del equivalent_dfs

    if missing_data_rows_dict:
        for missing_field, rows in missing_data_rows_dict.items():
            missing_data_df = pd.DataFrame(rows)
            missing_data_csv = os.path.join(cache_dir, f"{dataset_name}_missing_data_{missing_field}.csv")
            missing_data_df.to_csv(missing_data_csv, index=False)


if __name__ == "__main__":
    main()
