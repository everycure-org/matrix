# main.py
"""
Embedding Analysis Script

This script provides a template for analyzing embeddings using the Embedding Projector.
It includes functions for loading and caching embeddings and labels, as well as visualizing embeddings.

Usage:
    python main.py
"""

import os
import sys
import logging
import pandas as pd
import openai
import subprocess
from pathlib import Path
from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# =============================================================================
# Setup
# =============================================================================


def setup_environment(utils_path: str, root_subdir: str = "pipelines/matrix"):
    """
    Set up the Python environment by adding necessary paths and changing the working directory.

    Args:
        utils_path (str): Absolute path to the utilities directory.
        root_subdir (str): Subdirectory within the root path to change into.
    """
    if utils_path not in sys.path:
        sys.path.append(utils_path)
        logging.info(f"Added '{utils_path}' to sys.path.")

    try:
        root_path = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
        target_path = Path(root_path) / root_subdir
        os.chdir(target_path)
        logging.info(f"Changed working directory to '{target_path}'.")
    except subprocess.CalledProcessError:
        logging.error("Failed to get the root path using git. Ensure you're inside a git repository.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error during environment setup: {e}")
        sys.exit(1)


def load_embedding_utils():
    """
    Import necessary functions from embedding_utils.py.
    """
    try:
        from embedding_utils import (
            process_model,
            process_models,
            embedding_models_info,
            parse_list_string,
            load_datasets,
            load_embeddings_and_labels,
            missing_data_rows_dict,
            generate_candidate_pairs,
            refine_candidate_mappings_with_llm,
            find_additional_mappings_with_curategpt,
        )

        logging.info("Successfully imported embedding_utils functions.")
        return {
            "process_model": process_model,
            "process_models": process_models,
            "embedding_models_info": embedding_models_info,
            "parse_list_string": parse_list_string,
            "load_datasets": load_datasets,
            "load_embeddings_and_labels": load_embeddings_and_labels,
            "missing_data_rows_dict": missing_data_rows_dict,
            "generate_candidate_pairs": generate_candidate_pairs,
            "refine_candidate_mappings_with_llm": refine_candidate_mappings_with_llm,
            "find_additional_mappings_with_curategpt": find_additional_mappings_with_curategpt,
        }
    except ImportError as e:
        logging.error(f"Failed to import embedding_utils: {e}")
        sys.exit(1)


# =============================================================================
# Configuration
# =============================================================================


def configure_logging():
    """
    Configure the logging settings.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)]
    )
    logging.info("Logging is configured.")


def set_up_variables(root_path: str):
    """
    Set up configuration variables for the embedding analysis.

    Args:
        root_path (str): The root directory of the project.

    Returns:
        dict: A dictionary containing all configuration variables.
    """
    cache_dir = os.path.join(root_path, "apps", "embed_norm", "cached_datasets")
    os.makedirs(cache_dir, exist_ok=True)
    logging.info(f"Cache directory set at '{cache_dir}'.")

    for subdir in ["categories", "embeddings", "datasets"]:
        subdir_path = os.path.join(cache_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        logging.info(f"Subdirectory '{subdir}' created at '{subdir_path}'.")

    pos_seed = 54321
    neg_seed = 67890

    dataset_name = "rtx_kg2.int"
    nodes_dataset_name = "integration.int.rtx.nodes"
    edges_dataset_name = "integration.int.rtx.edges"

    categories = ["All Categories"]

    model_name = "OpenAI"
    model_names = ["OpenAI", "PubMedBERT", "SapBERT", "BlueBERT", "BioBERT"]

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logging.error("OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)
    openai.api_key = openai_api_key
    logging.info("OpenAI API key is set.")

    total_sample_size = 1000
    positive_ratio = 0.2

    positive_n = int(total_sample_size * positive_ratio)
    negative_n = total_sample_size - positive_n
    cache_suffix = f"_pos_{positive_n}_neg_{negative_n}"

    config = {
        "cache_dir": cache_dir,
        "pos_seed": pos_seed,
        "neg_seed": neg_seed,
        "dataset_name": dataset_name,
        "nodes_dataset_name": nodes_dataset_name,
        "edges_dataset_name": edges_dataset_name,
        "categories": categories,
        "model_name": model_name,
        "model_names": model_names,
        "total_sample_size": total_sample_size,
        "positive_ratio": positive_ratio,
        "positive_n": positive_n,
        "negative_n": negative_n,
        "cache_suffix": cache_suffix,
    }

    logging.info("Configuration variables are set.")
    return config


# =============================================================================
# Data Loading
# =============================================================================


def load_data(config: dict, embedding_utils: dict):
    """
    Load datasets and prepare for embedding processing.

    Args:
        config (dict): Configuration variables.
        embedding_utils (dict): Imported embedding_utils functions.

    Returns:
        tuple: (categories, positive_datasets, datasets, nodes_df)
    """
    configure_project("matrix")
    logging.info("Kedro project 'matrix' configured.")

    try:
        with KedroSession.create() as session:
            context = session.load_context()
            catalog = context.catalog
            nodes_df = catalog.load(config["nodes_dataset_name"])
            logging.info(f"Loaded nodes dataset '{config['nodes_dataset_name']}'.")
    except Exception as e:
        logging.error(f"Failed to load datasets using Kedro: {e}")
        sys.exit(1)

    try:
        categories, positive_datasets, datasets = embedding_utils["load_datasets"](
            nodes_df=nodes_df,
            cache_dir=os.path.join(config["cache_dir"], "datasets"),
            dataset_name=config["dataset_name"],
            pos_seed=config["pos_seed"],
            neg_seed=config["neg_seed"],
            total_sample_size=config["total_sample_size"],
            positive_ratio=config["positive_ratio"],
        )
        logging.info("Datasets and categories loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load datasets and categories: {e}")
        sys.exit(1)

    return categories, positive_datasets, datasets, nodes_df


# =============================================================================
# Define Functions
# =============================================================================


def node_to_string(row, text_fields, parse_list_string_func):
    """
    Convert a node row to a text string based on specified text fields.

    Args:
        row (pd.Series): A row from the nodes dataframe.
        text_fields (list): List of text fields to include in the representation.
        parse_list_string_func (function): Function to parse list-like strings.

    Returns:
        str: Concatenated text representation.
    """
    fields = [row.get(field, "") for field in text_fields]
    text_values = []
    for field_value in fields:
        if pd.notnull(field_value):
            parsed_list = parse_list_string_func(field_value)
            text_values.extend(parsed_list)
    return " ".join(text_values).strip()


def label_func(row):
    """
    Generate a custom label for a row.

    Args:
        row (pd.Series): A row from the nodes dataframe.

    Returns:
        str: Custom label string.
    """
    return f"{row['id']}, {row['name']}, custom label"


# =============================================================================
# Process Models
# =============================================================================


def generate_embeddings(config: dict, embedding_utils: dict, nodes_df: pd.DataFrame, datasets, positive_datasets):
    """
    Generate or load embeddings for both positive and negative datasets.

    Args:
        config (dict): Configuration variables.
        embedding_utils (dict): Imported embedding_utils functions.
        nodes_df (pd.DataFrame): The nodes dataframe.
        datasets (list): List of negative datasets.
        positive_datasets (list): List of positive datasets.

    Returns:
        tuple: (embeddings_dict, embeddings_dict_single)
    """
    # text_fields = ['name', 'description', 'category', 'labels', 'all_categories', 'equivalent_identifiers']
    text_fields = ["name", "category", "labels", "all_categories"]

    try:
        embeddings_dict = embedding_utils["process_models"](
            model_names=config["model_names"],
            positive_datasets=positive_datasets,
            negative_datasets=datasets,
            cache_dir=os.path.join(config["cache_dir"], "embeddings"),
            seed=config["neg_seed"],
            text_fields=text_fields,
            label_generation_func=lambda row: label_func(row),
            dataset_name=config["dataset_name"],
            use_ontogpt=False,
            cache_suffix=config["cache_suffix"],
            use_combinations=False,
            combine_fields=False,
        )
        logging.info("Embeddings for all models processed successfully using process_models().")
    except Exception as e:
        logging.error(f"Failed to process embeddings using process_models(): {e}")
        sys.exit(1)
    embeddings_dict_single = {}
    # try:
    #     embeddings_dict_single, _ = embedding_utils['process_model'](
    #         model_name=config['model_name'],
    #         model_info=embedding_utils['embedding_models_info'][config['model_name']],
    #         datasets=datasets,
    #         cache_dir=os.path.join(config['cache_dir'], 'embeddings'),
    #         seed=config['neg_seed'],
    #         text_fields=text_fields,
    #         text_representation_func=lambda row: node_to_string(row, text_fields, embedding_utils['parse_list_string']),
    #         label_generation_func=lambda row: label_func(row),
    #         dataset_name=config['dataset_name'],
    #         use_ontogpt=False,
    #         cache_suffix=config['cache_suffix'],
    #         use_combinations=False,
    #         combine_fields=False,
    #         dataset_type="negative"
    #     )
    #     logging.info("Embeddings for single model processed successfully using process_model().")
    # except Exception as e:
    #     logging.error(f"Failed to process embeddings using process_model(): {e}")
    #     sys.exit(1)

    return embeddings_dict, embeddings_dict_single


# =============================================================================
# Visualization
# =============================================================================


def visualize_embeddings(embeddings, category: str):
    """
    Visualize embeddings using PCA and matplotlib.

    Args:
        embeddings (numpy.ndarray): The embeddings to visualize.
        category (str): The category name for the plot title.
    """
    try:
        reduced_embeddings = PCA(n_components=2).fit_transform(embeddings)
        plt.figure(figsize=(10, 10))
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5)
        plt.title(f"Embeddings Visualization for {category}")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.grid(True)
        plt.savefig(f"embeddings_visualization_{category}.png")
        plt.show()
        logging.info(f"Embeddings for category '{category}' visualized successfully.")
    except Exception as e:
        logging.error(f"Failed to visualize embeddings: {e}")


# =============================================================================
# Main Function
# =============================================================================


def main():
    # Configure logging first to capture all logs
    configure_logging()

    # Setup environment paths
    utils_path = os.path.abspath("/home/wadmin/embed_norm/apps/embed_norm/embed_norm_test")
    setup_environment(utils_path=utils_path)

    # Import embedding utilities
    embedding_utils = load_embedding_utils()

    # Set up configuration variables
    root_path = Path(os.getcwd()).parents[1]  # Assuming current dir is 'pipelines/matrix'
    config = set_up_variables(root_path=str(root_path))

    # Load datasets and embeddings
    categories, positive_datasets, datasets, nodes_df = load_data(config, embedding_utils)

    # Generate or load embeddings
    embeddings_dict, embeddings_dict_single = generate_embeddings(
        config, embedding_utils, nodes_df, datasets, positive_datasets
    )

    # # Visualization for process_models()
    # category = 'All Categories'
    # model_key = f"{config['model_name']}_negative"
    # if category in embeddings_dict.get(model_key, {}):
    #     embeddings = embeddings_dict[model_key][category]
    #     visualize_embeddings(embeddings, category)
    # else:
    #     logging.error(f"Category '{category}' not found in embeddings_dict for process_models().")

    # # Visualization for process_model()
    # if category in embeddings_dict_single:
    #     embeddings_single = embeddings_dict_single[category]
    #     visualize_embeddings(embeddings_single, f"{category}_single_model")
    # else:
    #     logging.error(f"Category '{category}' not found in embeddings_dict_single for process_model().")


if __name__ == "__main__":
    main()
