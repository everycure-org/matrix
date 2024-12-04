# main.py

import sys
import os
import logging
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union, Optional
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import asyncio
import aiohttp
import joblib
from dotenv import load_dotenv
import openai
from openai import AsyncOpenAI
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
from pyspark.sql import DataFrame as SparkDataFrame
from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession
import json
import ast
from itertools import product
import tiktoken
import gc

missing_data_rows_dict = {}


def setup_environment(utils_path: Path, root_subdir: str = "pipelines/matrix"):
    if str(utils_path) not in sys.path:
        sys.path.append(str(utils_path))
        logging.info(f"Added '{utils_path}' to sys.path.")
    try:
        root_path = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip())
        target_path = root_path / root_subdir
        os.chdir(target_path)
        load_dotenv(dotenv_path=target_path / ".env")
        logging.info(f"Changed working directory to '{target_path}'.")
    except subprocess.CalledProcessError:
        logging.error("Failed to get the root path using git. Ensure you're inside a git repository.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error during environment setup: {e}")
        sys.exit(1)


def configure_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.StreamHandler(sys.stdout)]
    )
    logging.info("Logging is configured.")


class CacheManager:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_data(self, data: Any, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(data, file_path)
        logging.info(f"Data cached at {file_path}")

    def load_cached_data(self, file_path: Path):
        if file_path.exists():
            data = joblib.load(file_path)
            logging.info(f"Loaded cached data from {file_path}")
            return data
        else:
            logging.info(f"No cache found at {file_path}")
            return None

    def get_or_compute(self, cache_path: Union[str, Path], compute_func: Callable, *args, **kwargs):
        cache_path = Path(cache_path)
        result = self.load_cached_data(cache_path)
        if result is not None:
            return result
        else:
            result = compute_func(*args, **kwargs)
            self.cache_data(result, cache_path)
            return result


@dataclass
class Config:
    cache_dir: Path
    pos_seed: int
    neg_seed: int
    dataset_name: str
    nodes_dataset_name: str
    edges_dataset_name: str
    categories: List[str]
    model_names: List[str]
    total_sample_size: int
    positive_ratio: float
    positive_n: int
    negative_n: int
    cache_suffix: str
    embedding_models_info: dict = field(
        default_factory=lambda: {
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
    )
    use_llm_enhancement: bool = False
    llm_prompt_template: str = (
        "What biological entity does this information represent? "
        "Please be descriptive in such a way that an embedding of the text "
        "would enable identification of similar entities: {row_data}"
    )


class DataLoader:
    def __init__(self, cache_manager: CacheManager, config: Config):
        self.cache_manager = cache_manager
        self.config = config

    def load_datasets(
        self, nodes_df
    ) -> Tuple[List[str], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame]:
        if isinstance(nodes_df, SparkDataFrame):
            nodes_df = nodes_df.toPandas()
        categories_cache_path = self.cache_manager.cache_dir / "datasets" / f"{self.config.dataset_name}_categories.pkl"
        categories = self.cache_manager.get_or_compute(categories_cache_path, self._compute_categories, nodes_df)
        positive_datasets = {}
        negative_datasets = {}
        for category in categories:
            llm_suffix = "_llm" if self.config.use_llm_enhancement else ""
            positive_cache_path = (
                self.cache_manager.cache_dir
                / "datasets"
                / f"{self.config.dataset_name}_sampled_df_positives_{category}_seed_{self.config.pos_seed}{self.config.cache_suffix}{llm_suffix}.pkl"
            )
            negative_cache_path = (
                self.cache_manager.cache_dir
                / "datasets"
                / f"{self.config.dataset_name}_sampled_df_negatives_{category}_seed_{self.config.neg_seed}{self.config.cache_suffix}{llm_suffix}.pkl"
            )

            def compute_positive_dataset():
                df = self._sample_positive_dataset(nodes_df, category)
                normalizer = Normalizer(self.cache_manager.cache_dir)
                df = normalizer.augment_positive_df(df)
                if self.config.use_llm_enhancement:
                    llm_enhancer = LLMEnhancer(self.config, self.cache_manager)
                    df = llm_enhancer.augment_dataframe(df)
                return df

            def compute_negative_dataset():
                df = self._sample_negative_dataset(nodes_df, category)
                if self.config.use_llm_enhancement:
                    llm_enhancer = LLMEnhancer(self.config, self.cache_manager)
                    df = llm_enhancer.augment_dataframe(df)
                return df

            positive_df = self.cache_manager.get_or_compute(positive_cache_path, compute_positive_dataset)
            negative_df = self.cache_manager.get_or_compute(negative_cache_path, compute_negative_dataset)

            positive_datasets[category] = positive_df
            negative_datasets[category] = negative_df
        return categories, positive_datasets, negative_datasets, nodes_df

    def _compute_categories(self, nodes_df: pd.DataFrame) -> List[str]:
        if isinstance(nodes_df, SparkDataFrame):
            nodes_df = nodes_df.toPandas()
        categories = nodes_df["category"].unique().tolist()
        categories.append("All Categories")
        return categories

    def _sample_positive_dataset(self, nodes_df: pd.DataFrame, category: str) -> pd.DataFrame:
        if isinstance(nodes_df, SparkDataFrame):
            nodes_df = nodes_df.toPandas()
        if category == "All Categories":
            category_df = nodes_df.copy()
        else:
            category_df = nodes_df[nodes_df["category"] == category]
        positive_df = category_df.sample(
            n=min(self.config.positive_n, len(category_df)), random_state=self.config.pos_seed
        )
        return positive_df

    def _sample_negative_dataset(self, nodes_df: pd.DataFrame, category: str) -> pd.DataFrame:
        if isinstance(nodes_df, SparkDataFrame):
            nodes_df = nodes_df.toPandas()
        if category == "All Categories":
            negative_df = nodes_df.copy()
        else:
            negative_df = nodes_df[nodes_df["category"] == category]
        negative_df = negative_df.sample(
            n=min(self.config.negative_n, len(negative_df)), random_state=self.config.neg_seed
        )
        return negative_df


def parse_list_string(s):
    if isinstance(s, list):
        return s
    elif isinstance(s, (np.ndarray, pd.Series)):
        return s.tolist()
    elif isinstance(s, str):
        s = s.strip()
        if not s or s == "[]":
            return []
        if s.startswith("[") and s.endswith("]"):
            s_json = s.replace("'", '"')
            try:
                return json.loads(s_json)
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(s)
                except (ValueError, SyntaxError):
                    pass
        return [s]
    else:
        return []


def is_missing_value(value):
    if isinstance(value, (list, np.ndarray, pd.Series)):
        return pd.isnull(value).all() or all(not str(v).strip() for v in value)
    else:
        return pd.isnull(value) or not str(value).strip()


def get_text_representation(row, text_fields=None, combine_fields=False):
    if text_fields is None:
        text_fields = ["name", "category", "labels", "all_categories"]
    if "llm_enhanced_text" in row and pd.notnull(row["llm_enhanced_text"]):
        text_fields = ["llm_enhanced_text"]
    global missing_data_rows_dict
    fields = [row.get(field, "") for field in text_fields]
    missing_fields = [field for field, value in zip(text_fields, fields) if is_missing_value(value)]
    for missing_field in missing_fields:
        if missing_field not in missing_data_rows_dict:
            missing_data_rows_dict[missing_field] = []
        missing_data_rows_dict[missing_field].append(row)
    if combine_fields:
        parsed_lists = [parse_list_string(field_value) for field_value in fields]
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


def batch_texts_by_token_limit(texts, max_tokens_per_request=8191, model_name="text-embedding-3-small"):
    encoding = tiktoken.get_encoding("cl100k_base")
    batches = []
    current_batch = []
    current_tokens = 0
    for text in texts:
        tokens = encoding.encode(text)
        num_tokens = len(tokens)
        if num_tokens > 8191:
            logging.warning("Text exceeds max tokens per text (8191), truncating.")
            tokens = tokens[:8191]
            text = encoding.decode(tokens)
            num_tokens = 8191
        if current_tokens + num_tokens > max_tokens_per_request:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
        current_batch.append(text)
        current_tokens += num_tokens
    if current_batch:
        batches.append(current_batch)
    return batches


def create_equivalent_items_dfs(positive_datasets, normalized_data):
    equivalent_dfs = {}
    for category, df in tqdm(
        positive_datasets.items(),
        total=len(positive_datasets),
        desc="Creating equivalent items DataFrames",
        leave=False,
    ):
        if df.empty:
            logging.warning(f"The DataFrame for category '{category}' is empty. Skipping.")
            equivalent_dfs[category] = df
            continue
        if "id" not in df.columns:
            logging.error(
                f"The 'id' column is missing in DataFrame for category '{category}'. Columns: {df.columns.tolist()}"
            )
            equivalent_dfs[category] = df
            continue
        df["ids_to_normalize"] = df.apply(
            lambda row: list(set([row["id"]] + parse_list_string(row.get("equivalent_identifiers", [])))), axis=1
        )
        df_exploded = df.explode("ids_to_normalize").reset_index(drop=True)
        df_exploded.rename(columns={"ids_to_normalize": "id_to_normalize"}, inplace=True)
        unique_ids = df_exploded["id_to_normalize"].unique()
        equivalents_list = []
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
        merged_df = df_exploded.merge(equivalents_df, on="id_to_normalize", how="left")
        merged_df.rename(columns={"id_to_normalize": "ids_to_normalize"}, inplace=True)
        columns_to_select = df.columns
        for col in columns_to_select:
            if col in merged_df.columns:
                if merged_df[col].apply(lambda x: isinstance(x, (list, dict, set, np.ndarray))).any():
                    merged_df[col] = merged_df[col].apply(lambda x: str(x) if not isinstance(x, str) else x)
            else:
                logging.warning(f"Column '{col}' not found in merged_df. Skipping this column.")
        equivalent_df = merged_df[columns_to_select].drop_duplicates()
        equivalent_dfs[category] = equivalent_df
    return equivalent_dfs


class Normalizer:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.MAX_RETRIES = 3
        self.BATCH_SIZE = 100
        self.NORM_URL = "https://nodenorm.transltr.io/1.5/get_normalized_nodes"
        self.normalized_data = {}
        self.failed_ids = set()

    async def normalize_node(self, session, url, payload):
        retries = 0
        while retries < self.MAX_RETRIES:
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

    async def batch_normalize_curies_async(self, category_curies: Dict[str, List[str]]):
        all_curies = [curie for curies in category_curies.values() for curie in curies]
        total_batches = (len(all_curies) + self.BATCH_SIZE - 1) // self.BATCH_SIZE
        semaphore = asyncio.Semaphore(10)

        async def sem_normalize(session, url, payload):
            async with semaphore:
                return await self.normalize_node(session, url, payload)

        conn = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=conn) as session:
            tasks = []
            with tqdm(total=total_batches, desc="Normalizing nodes", leave=False) as pbar:
                for i in range(0, len(all_curies), self.BATCH_SIZE):
                    batch = all_curies[i : i + self.BATCH_SIZE]
                    payload = {"curies": batch, "conflate": False, "expand_all": True}
                    task = asyncio.ensure_future(sem_normalize(session, self.NORM_URL, payload))
                    tasks.append(task)
                    pbar.update(1)
            responses = await asyncio.gather(*tasks, return_exceptions=True)
        for response in tqdm(responses, desc="Processing normalization responses", leave=False):
            if isinstance(response, Exception):
                logging.error(f"Normalization request resulted in an exception: {response}")
                continue
            if response:
                for key, value in response.items():
                    if isinstance(value, dict):
                        equivalents = value.get("equivalent_identifiers", [])
                        value_type = value.get("type", [])
                        if isinstance(value_type, str):
                            types = [value_type.strip('"')]
                        elif isinstance(value_type, list):
                            types = [t.strip('"') for t in value_type]
                        else:
                            types = []
                        normalized_equivalents = [
                            {"identifier": eq.get("identifier"), "label": eq.get("label", ""), "types": types}
                            for eq in equivalents
                            if eq.get("identifier")
                        ]
                        if normalized_equivalents:
                            self.normalized_data[key] = normalized_equivalents
                        else:
                            self.failed_ids.add(key)
                    else:
                        self.failed_ids.add(key)

    def augment_positive_df(self, positive_df: pd.DataFrame) -> pd.DataFrame:
        if positive_df.empty:
            logging.warning("Positive DataFrame is empty. Skipping augmentation.")
            return positive_df
        category_curies = {"positive": positive_df["id"].tolist()}
        asyncio.run(self.batch_normalize_curies_async(category_curies))
        equivalent_dfs = create_equivalent_items_dfs({"positive": positive_df}, self.normalized_data)
        augmented_positive_df = equivalent_dfs["positive"]
        return augmented_positive_df


class EmbeddingGenerator:
    def __init__(self, config: Config, cache_manager: CacheManager):
        self.config = config
        self.cache_manager = cache_manager
        self.embedding_models_info = config.embedding_models_info
        self.MAX_RETRIES = 3
        self.BATCH_SIZE = 100
        self.MAX_TOKENS_PER_TEXT = 8191
        self.MAX_TOKENS_PER_REQUEST = 8191
        self.TOKENIZER_NAME = "text-embedding-3-small"
        self.TOKENIZER_ENCODING = "cl100k_base"
        self.client = AsyncOpenAI()

    def load_model_and_tokenizer(self, model_info):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_info["tokenizer_name"])
        model = AutoModel.from_pretrained(model_info["model_name"])
        model.to(device)
        model.eval()
        return model, tokenizer

    def unload_model_and_tokenizer(self, model, tokenizer):
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def compute_embeddings_hf(self, model, tokenizer, texts, batch_size=16, max_length=512):
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

    async def get_openai_embedding_async(self, texts: List[str]) -> Optional[np.ndarray]:
        retries = 0
        while retries < self.MAX_RETRIES:
            try:
                response = await self.client.embeddings.create(input=texts, model=self.TOKENIZER_NAME)
                embeddings = [item["embedding"] for item in response.model_dump()["data"]]
                return np.array(embeddings)
            except openai.RateLimitError as e:
                retries += 1
                logging.warning(f"Rate limit error: {e}. Retrying {retries}/{self.MAX_RETRIES} after delay.")
                await asyncio.sleep(2**retries)
            except openai.APIError as e:
                logging.error(f"Invalid request: {e}. Skipping batch.")
                return None
            except openai.APIConnectionError as e:
                retries += 1
                logging.error(f"Server could not be reached: {e}. Retrying {retries}/{self.MAX_RETRIES}.")
                await asyncio.sleep(2**retries)
            except Exception as e:
                retries += 1
                logging.error(f"Unexpected error: {e}. Retrying {retries}/{self.MAX_RETRIES}.")
                await asyncio.sleep(2**retries)
        logging.error("Max retries exceeded for get_openai_embedding_async.")
        return None

    async def get_openai_embeddings_in_batches_async(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        batches = batch_texts_by_token_limit(texts, self.MAX_TOKENS_PER_REQUEST, self.TOKENIZER_NAME)
        tasks = [self.get_openai_embedding_async(batch) for batch in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.warning(f"Batch {i} resulted in an exception: {result}. Skipping.")
                continue
            if result is not None:
                embeddings.append(result)
        if embeddings:
            return np.vstack(embeddings)
        logging.error("No embeddings were computed.")
        return np.array([])

    def get_openai_embeddings_in_batches(self, texts):
        return asyncio.run(self.get_openai_embeddings_in_batches_async(texts))

    def process_model(
        self,
        model_name: str,
        model_info: Dict[str, Any],
        datasets: Dict[str, pd.DataFrame],
        seed: int,
        text_fields: List[str] = None,
        label_generation_func=None,
        dataset_name: str = "default",
        use_ontogpt: bool = False,
        cache_suffix: str = "",
        use_combinations: bool = False,
        combine_fields: bool = False,
        dataset_type: str = "negative",
    ) -> Tuple[str, Dict[str, np.ndarray]]:
        embeddings_dict = {}
        labels_dict = {}
        similarities_dict = {}
        combinations_suffix = "_combinations" if use_combinations else ""
        dataset_type_suffix = f"_{dataset_type}"
        if model_info["type"] == "hf":
            model, tokenizer = self.load_model_and_tokenizer(model_info)
        for category_name, df in tqdm(
            datasets.items(), desc=f"Processing model {model_name} ({dataset_type})", total=len(datasets), leave=False
        ):
            if df.empty:
                logging.warning(f"The DataFrame for category '{category_name}' is empty. Skipping.")
                continue
            cache_file = (
                self.cache_manager.cache_dir
                / "embeddings"
                / f"{dataset_name}{dataset_type_suffix}_embeddings_{category_name}_{model_name}{combinations_suffix}_seed_{seed}{cache_suffix}.pkl"
            )
            labels_file = (
                self.cache_manager.cache_dir
                / "embeddings"
                / f"{dataset_name}{dataset_type_suffix}_labels_{category_name}_{model_name}{combinations_suffix}_seed_{seed}{cache_suffix}.pkl"
            )
            sim_file = (
                self.cache_manager.cache_dir
                / "embeddings"
                / f"{dataset_name}{dataset_type_suffix}_cosine_similarities_{category_name}_{model_name}{combinations_suffix}_seed_{seed}{cache_suffix}.pkl"
            )

            def compute_embeddings():
                all_texts = df.apply(lambda row: get_text_representation(row, text_fields, False), axis=1).tolist()
                if not all_texts:
                    logging.warning(f"No texts to process for category '{category_name}'. Skipping.")
                    return np.array([])
                if model_info["type"] == "hf":
                    embeddings = self.compute_embeddings_hf(model, tokenizer, all_texts)
                elif model_info["type"] == "openai":
                    embeddings = self.get_openai_embeddings_in_batches(all_texts)
                else:
                    embeddings = np.array([])
                return embeddings

            embeddings = self.cache_manager.get_or_compute(cache_file, compute_embeddings)
            if embeddings.size == 0:
                logging.warning(
                    f"No embeddings computed for category '{category_name}'. Skipping similarity computation."
                )
                continue
            embeddings_dict[category_name] = embeddings

            def compute_labels():
                if label_generation_func is None:
                    labels = [
                        f"Row {idx}: {text}"
                        for idx, text in enumerate(
                            df.apply(lambda row: get_text_representation(row, text_fields, False), axis=1).tolist()
                        )
                    ]
                else:
                    labels = df.apply(label_generation_func, axis=1).tolist()
                return labels

            labels = self.cache_manager.get_or_compute(labels_file, compute_labels)
            labels_dict[category_name] = labels

            def compute_similarities():
                similarities = cosine_similarity(embeddings)
                return similarities

            similarities = self.cache_manager.get_or_compute(sim_file, compute_similarities)
            similarities_dict[category_name] = similarities
        if model_info["type"] == "hf":
            self.unload_model_and_tokenizer(model, tokenizer)
        return model_name, embeddings_dict

    def process_models(
        self,
        model_names: List[str],
        positive_datasets: Dict[str, pd.DataFrame],
        negative_datasets: Dict[str, pd.DataFrame],
        seed: int,
        text_fields: List[str] = None,
        label_generation_func=None,
        dataset_name: str = "default",
        use_ontogpt: bool = False,
        cache_suffix: str = "",
        use_combinations: bool = False,
        combine_fields: bool = False,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        embeddings_dict_all_models = {}
        for model_name in tqdm(model_names, desc="Processing models", leave=False):
            model_info = self.embedding_models_info.get(model_name)
            if not model_info:
                logging.warning(f"Model '{model_name}' not found in embedding_models_info. Skipping.")
                continue
            datasets_to_process = [
                ("positive", positive_datasets),
                ("negative", negative_datasets),
            ]
            for dataset_type, datasets in datasets_to_process:
                model_key = f"{model_name}_{dataset_type}"
                _, embeddings_dict = self.process_model(
                    model_name=model_name,
                    model_info=model_info,
                    datasets=datasets,
                    seed=seed,
                    text_fields=text_fields,
                    label_generation_func=label_generation_func,
                    dataset_name=dataset_name,
                    use_ontogpt=use_ontogpt,
                    cache_suffix=cache_suffix,
                    use_combinations=use_combinations,
                    combine_fields=combine_fields,
                    dataset_type=dataset_type,
                )
                embeddings_dict_all_models[model_key] = embeddings_dict
        return embeddings_dict_all_models


class LLMEnhancer:
    def __init__(self, config: Config, cache_manager: CacheManager):
        self.config = config
        self.cache_manager = cache_manager
        self.client = AsyncOpenAI()
        self.MAX_RETRIES = 3

    async def enhance_row(self, session, prompt: str) -> str:
        retries = 0
        while retries < self.MAX_RETRIES:
            try:
                response = await self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}], model="gpt-4o", temperature=0.7
                )
                return response.choices[0].message.content.strip()
            except openai.RateLimitError as e:
                retries += 1
                logging.warning(f"Rate limit error: {e}. Retrying {retries}/{self.MAX_RETRIES} after delay.")
                await asyncio.sleep(2**retries)
            except openai.OpenAIError as e:
                retries += 1
                logging.error(f"OpenAI error: {e}. Retrying {retries}/{self.MAX_RETRIES}.")
                await asyncio.sleep(2**retries)
            except Exception as e:
                retries += 1
                logging.error(f"Unexpected error: {e}. Retrying {retries}/{self.MAX_RETRIES}.")
                await asyncio.sleep(2**retries)
        logging.error("Max retries exceeded for enhance_row.")
        return ""

    async def augment_dataframe_async(self, df: pd.DataFrame) -> pd.DataFrame:
        prompts = []
        for idx, row in df.iterrows():
            row_data = row.to_dict()
            row_data_str = json.dumps(row_data, default=str)
            prompt = self.config.llm_prompt_template.format(row_data=row_data_str)
            prompts.append((idx, prompt))
        semaphore = asyncio.Semaphore(5)

        async def sem_enhance(session, idx, prompt):
            async with semaphore:
                result = await self.enhance_row(session, prompt)
                return idx, result

        conn = aiohttp.TCPConnector(limit=5)
        async with aiohttp.ClientSession(connector=conn) as session:
            tasks = []
            for idx, prompt in prompts:
                task = asyncio.ensure_future(sem_enhance(session, idx, prompt))
                tasks.append(task)
            results = await asyncio.gather(*tasks, return_exceptions=True)
        enhanced_texts = {}
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"Error during LLM augmentation: {result}")
                continue
            idx, enhanced_text = result
            enhanced_texts[idx] = enhanced_text
        df["llm_enhanced_text"] = df.index.map(enhanced_texts)
        return df

    def augment_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        return asyncio.run(self.augment_dataframe_async(df))


def set_up_variables(root_path: Path) -> Config:
    cache_dir = root_path / "apps" / "embed_norm" / "cached_datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Cache directory set at '{cache_dir}'.")
    for subdir in ["embeddings", "datasets"]:
        subdir_path = cache_dir / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Subdirectory '{subdir}' created at '{subdir_path}'.")
    pos_seed = 54321
    neg_seed = 67890
    dataset_name = "rtx_kg2.int"
    nodes_dataset_name = "integration.int.rtx.nodes"
    edges_dataset_name = "integration.int.rtx.edges"
    categories = ["All Categories"]
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
    config = Config(
        cache_dir=cache_dir,
        pos_seed=pos_seed,
        neg_seed=neg_seed,
        dataset_name=dataset_name,
        nodes_dataset_name=nodes_dataset_name,
        edges_dataset_name=edges_dataset_name,
        categories=categories,
        model_names=model_names,
        total_sample_size=total_sample_size,
        positive_ratio=positive_ratio,
        positive_n=positive_n,
        negative_n=negative_n,
        cache_suffix=cache_suffix,
        use_llm_enhancement=False,  # Set to True to enable LLM enhancement
    )
    logging.info("Configuration variables are set.")
    return config


def load_data(
    config: Config, cache_manager: CacheManager
) -> Tuple[List[str], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame]:
    configure_project("matrix")
    logging.info("Kedro project 'matrix' configured.")
    try:
        with KedroSession.create() as session:
            context = session.load_context()
            catalog = context.catalog
            nodes_df = catalog.load(config.nodes_dataset_name)
        if isinstance(nodes_df, SparkDataFrame):
            nodes_df = nodes_df.toPandas()
        data_loader = DataLoader(cache_manager, config)
        categories, positive_datasets, negative_datasets, nodes_df = data_loader.load_datasets(nodes_df=nodes_df)
        logging.info("Datasets and categories loaded successfully.")
        return categories, positive_datasets, negative_datasets, nodes_df
    except Exception as e:
        logging.error(f"Failed to load datasets using Kedro: {e}")
        sys.exit(1)


def label_func(row: pd.Series) -> str:
    idx = row.name
    text = row.get("text", "")
    return f"Row {idx}: {text}"


def generate_embeddings(
    config: Config,
    cache_manager: CacheManager,
    nodes_df: pd.DataFrame,
    negative_datasets: Dict[str, pd.DataFrame],
    positive_datasets: Dict[str, pd.DataFrame],
):
    if config.use_llm_enhancement:
        text_fields = ["llm_enhanced_text"]
    else:
        text_fields = ["name", "category", "labels", "all_categories"]
    embedding_generator = EmbeddingGenerator(config, cache_manager)
    embeddings_dict_all_models = embedding_generator.process_models(
        model_names=config.model_names,
        positive_datasets=positive_datasets,
        negative_datasets=negative_datasets,
        seed=config.neg_seed,
        text_fields=text_fields,
        label_generation_func=None,
        dataset_name=config.dataset_name,
        use_ontogpt=False,
        cache_suffix=config.cache_suffix,
        use_combinations=False,
        combine_fields=False,
    )
    logging.info("Embeddings for models processed successfully using process_models().")
    return embeddings_dict_all_models


def main():
    configure_logging()
    utils_path = Path(__file__).parent.resolve()
    setup_environment(utils_path=utils_path)
    root_path = Path.cwd().parents[1]
    config = set_up_variables(root_path=root_path)
    cache_manager = CacheManager(config.cache_dir)
    _, positive_datasets, negative_datasets, nodes_df = load_data(config, cache_manager)
    generate_embeddings(config, cache_manager, nodes_df, negative_datasets, positive_datasets)


if __name__ == "__main__":
    main()
