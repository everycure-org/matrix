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
import torch
from transformers import AutoTokenizer, AutoModel
from pyspark.sql import DataFrame as SparkDataFrame
from kedro.framework.session import KedroSession
from kedro.framework.project import configure_project
import json
import ast
import tiktoken
import gc

from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio


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
    embedding_models_info: Dict[str, Dict[str, Any]] = field(
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
    llm_prompt_template: str = (
        "What biological entity does this information represent? "
        "Please be descriptive in such a way that an embedding of the text "
        "would enable identification of similar entities: {row_data}"
    )


class Environment:
    @staticmethod
    def setup_environment(utils_path: Path, root_subdir: str = "pipelines/matrix"):
        if str(utils_path) not in sys.path:
            sys.path.append(str(utils_path))
        try:
            root_path = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip())
            target_path = root_path / root_subdir
            os.chdir(target_path)
            load_dotenv(dotenv_path=target_path / ".env")
        except subprocess.CalledProcessError:
            sys.exit(1)
        except Exception:
            sys.exit(1)

    @staticmethod
    def configure_logging():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
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
            try:
                data = joblib.load(file_path)
                logging.info(f"Loaded cached data from {file_path}")
                return data
            except Exception as e:
                logging.error(f"Failed to load cache from {file_path}: {e}")
                return None
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


class DataLoader:
    def __init__(self, cache_manager: CacheManager, config: Config, normalizer: "Normalizer"):
        self.cache_manager = cache_manager
        self.config = config
        self.normalizer = normalizer

    def load_datasets(
        self, nodes_df: pd.DataFrame
    ) -> Tuple[List[str], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame]:
        categories = self.get_categories(nodes_df)
        positive_datasets = {}
        negative_datasets = {}
        for category in tqdm(categories, desc="Processing categories"):
            positive_df = self.sample_positive_dataset(nodes_df, category)
            positive_df = self.normalizer.augment_positive_df(positive_df)
            negative_df = self.sample_negative_dataset(nodes_df, category)
            positive_datasets[category] = positive_df
            negative_datasets[category] = negative_df
        return categories, positive_datasets, negative_datasets, nodes_df

    def get_categories(self, nodes_df: pd.DataFrame) -> List[str]:
        categories = nodes_df["category"].unique().tolist()
        categories.append("All Categories")
        return categories

    def sample_positive_dataset(self, nodes_df: pd.DataFrame, category: str) -> pd.DataFrame:
        if category == "All Categories":
            category_df = nodes_df.copy()
        else:
            category_df = nodes_df[nodes_df["category"] == category]
        return category_df.sample(
            n=min(self.config.positive_n, len(category_df)),
            random_state=self.config.pos_seed,
        ).reset_index(drop=True)

    def sample_negative_dataset(self, nodes_df: pd.DataFrame, category: str) -> pd.DataFrame:
        if category == "All Categories":
            negative_df = nodes_df.copy()
        else:
            negative_df = nodes_df[nodes_df["category"] != category]
        return negative_df.sample(
            n=min(self.config.negative_n, len(negative_df)),
            random_state=self.config.neg_seed,
        ).reset_index(drop=True)


class Normalizer:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_file = self.cache_dir / "normalized_data.pkl"
        self.MAX_RETRIES = 3
        self.BATCH_SIZE = 100
        self.NORM_URL = "https://nodenorm.transltr.io/1.5/get_normalized_nodes"
        self.normalized_data = self.load_normalized_cache()
        self.failed_ids = set()

    def load_normalized_cache(self) -> Dict[str, Any]:
        if self.cache_file.exists():
            try:
                return joblib.load(self.cache_file)
            except Exception:
                return {}
        else:
            return {}

    def save_normalized_cache(self):
        try:
            joblib.dump(self.normalized_data, self.cache_file)
        except Exception:
            pass

    async def normalize_node(
        self, session: aiohttp.ClientSession, url: str, payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
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
        curies_to_normalize = [curie for curie in all_curies if curie not in self.normalized_data]
        if not curies_to_normalize:
            return
        semaphore = asyncio.Semaphore(10)

        async def sem_normalize(session, url, payload):
            async with semaphore:
                return await self.normalize_node(session, url, payload)

        conn = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=conn) as session:
            tasks = []
            for i in range(0, len(curies_to_normalize), self.BATCH_SIZE):
                batch = curies_to_normalize[i : i + self.BATCH_SIZE]
                payload = {"curies": batch, "conflate": False, "expand_all": True}
                task = asyncio.ensure_future(sem_normalize(session, self.NORM_URL, payload))
                tasks.append(task)
            responses = []
            for future in tqdm_asyncio.as_completed(tasks, desc="Normalizing nodes", total=len(tasks)):
                response = await future
                responses.append(response)
        for response in responses:
            if isinstance(response, Exception):
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
                            {
                                "id": eq.get("identifier"),
                                "name": eq.get("label", ""),
                                "types": types,
                            }
                            for eq in equivalents
                            if eq.get("identifier")
                        ]
                        if normalized_equivalents:
                            self.normalized_data[key] = normalized_equivalents
                        else:
                            self.failed_ids.add(key)
                    else:
                        self.failed_ids.add(key)
        self.save_normalized_cache()

    def augment_positive_df(self, positive_df: pd.DataFrame) -> pd.DataFrame:
        if positive_df.empty:
            return positive_df
        category_curies = {"positive": positive_df["id"].tolist()}
        asyncio.run(self.batch_normalize_curies_async(category_curies))
        equivalent_dfs = self.create_equivalent_items_dfs({"positive": positive_df}, self.normalized_data)
        augmented_positive_df = equivalent_dfs["positive"]
        return augmented_positive_df

    @staticmethod
    def create_equivalent_items_dfs(
        positive_datasets: Dict[str, pd.DataFrame],
        normalized_data: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, pd.DataFrame]:
        equivalent_dfs = {}
        for category, df in positive_datasets.items():
            if df.empty:
                equivalent_dfs[category] = df
                continue
            if "id" not in df.columns:
                equivalent_dfs[category] = df
                continue
            df["ids_to_normalize"] = df.apply(
                lambda row: list(
                    set(
                        [str(row["id"])]
                        + [str(x) for x in Normalizer.parse_list_string(row.get("equivalent_identifiers", []))]
                    )
                ),
                axis=1,
            )
            df_exploded = df.explode("ids_to_normalize").reset_index(drop=True)
            df_exploded.rename(columns={"ids_to_normalize": "id_to_normalize"}, inplace=True)

            equivalents_list = []
            exploded_records = df_exploded.to_dict(orient="records")
            for row in exploded_records:
                id_value = row["id_to_normalize"]
                if id_value in normalized_data:
                    equivalents = normalized_data[id_value]
                    for eq in equivalents:
                        new_row = row.copy()
                        new_row["id"] = eq["id"]
                        new_row["name"] = eq["name"]
                        new_row["all_categories"] = eq.get("types", [])
                        equivalents_list.append(new_row)
                else:
                    equivalents_list.append(row)

            equivalent_df = pd.DataFrame(equivalents_list)

            for col in equivalent_df.columns:
                if equivalent_df[col].apply(lambda x: isinstance(x, (list, dict, set, np.ndarray))).any():
                    equivalent_df[col] = equivalent_df[col].apply(safe_json_dumps)

            equivalent_df = equivalent_df.drop_duplicates()
            equivalent_dfs[category] = equivalent_df
        return equivalent_dfs

    @staticmethod
    def parse_list_string(s: Any) -> List[str]:
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


class EmbeddingGenerator:
    def __init__(self, config: Config, cache_manager: CacheManager):
        self.config = config
        self.cache_manager = cache_manager
        self.embedding_models_info = config.embedding_models_info
        self.MAX_RETRIES = 3
        self.BATCH_SIZE = 100
        self.MAX_TOKENS_PER_TEXT = 8191
        self.MAX_TOKENS_PER_REQUEST = 8191
        self.TOKENIZER_NAME = "text-embedding-ada-002"
        self.TOKENIZER_ENCODING = "cl100k_base"
        self.client = AsyncOpenAI()

    def load_model_and_tokenizer(self, model_info: Dict[str, Any]) -> Tuple[AutoModel, AutoTokenizer]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_info["tokenizer_name"])
        model = AutoModel.from_pretrained(model_info["model_name"])
        model.to(device)
        model.eval()
        return model, tokenizer

    def unload_model_and_tokenizer(self, model: AutoModel, tokenizer: AutoTokenizer):
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def compute_embeddings_hf(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        texts: List[str],
        batch_size: int = 16,
        max_length: int = 512,
    ) -> np.ndarray:
        if not texts:
            return np.array([])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embeddings = []
        dataset = list(zip(texts))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        for batch in tqdm(dataloader, desc="Computing embeddings"):
            batch_texts = batch[0]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeddings)
        if embeddings:
            embeddings = np.vstack(embeddings)
        else:
            embeddings = np.array([])
        return embeddings

    async def get_openai_embedding_async(self, texts: List[str]) -> Optional[np.ndarray]:
        if not texts or all(not text.strip() for text in texts):
            logging.warning("Empty or whitespace-only texts provided to OpenAI embeddings.")
            return np.array([])
        retries = 0
        while retries < self.MAX_RETRIES:
            try:
                response = await self.client.embeddings.create(input=texts, model=self.TOKENIZER_NAME)
                embeddings = [item["embedding"] for item in response.model_dump()["data"]]
                return np.array(embeddings)
            except openai.RateLimitError:
                retries += 1
                await asyncio.sleep(2**retries)
            except openai.APIError:
                return None
            except openai.APIConnectionError:
                retries += 1
                await asyncio.sleep(2**retries)
            except Exception:
                retries += 1
                await asyncio.sleep(2**retries)
        return None

    async def get_openai_embeddings_in_batches_async(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        batches = self.batch_texts_by_token_limit(texts, self.MAX_TOKENS_PER_REQUEST, self.TOKENIZER_NAME)
        for batch in tqdm(batches, desc="Getting embeddings from OpenAI"):
            result = await self.get_openai_embedding_async(batch)
            if result is not None and result.size > 0:
                embeddings.append(result)
            else:
                logging.warning("No embeddings returned for a batch.")
        if embeddings:
            return np.vstack(embeddings)
        return np.array([])

    def get_openai_embeddings_in_batches(self, texts: List[str]) -> np.ndarray:
        return asyncio.run(self.get_openai_embeddings_in_batches_async(texts))

    @staticmethod
    def batch_texts_by_token_limit(
        texts: List[str],
        max_tokens_per_request: int = 8191,
        model_name: str = "text-embedding-ada-002",
    ) -> List[List[str]]:
        encoding = tiktoken.get_encoding("cl100k_base")
        batches = []
        current_batch = []
        current_tokens = 0
        for text in texts:
            tokens = encoding.encode(text)
            num_tokens = len(tokens)
            if num_tokens > 8191:
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

    def process_model(
        self,
        model_name: str,
        model_info: Dict[str, Any],
        datasets: Dict[str, pd.DataFrame],
        seed: int,
        text_fields: List[str],
        dataset_name: str = "default",
        cache_suffix: str = "",
        dataset_type: str = "negative",
    ) -> Tuple[str, Dict[str, np.ndarray]]:
        embeddings_dict = {}
        if model_info["type"] == "hf":
            model, tokenizer = self.load_model_and_tokenizer(model_info)
        for category_name, df in datasets.items():
            if df.empty:
                continue
            dataset_type_suffix = f"_{dataset_type}"
            cache_file = (
                self.cache_manager.cache_dir
                / "embeddings"
                / f"{dataset_name}{dataset_type_suffix}_embeddings_{category_name}_{model_name}_seed_{seed}{cache_suffix}.pkl"
            )
            ids_file = (
                self.cache_manager.cache_dir
                / "embeddings"
                / f"{dataset_name}{dataset_type_suffix}_ids_{category_name}_{model_name}_seed_{seed}{cache_suffix}.pkl"
            )

            all_texts = [self.get_text_representation(row, text_fields) for _, row in df.iterrows()]
            if not all_texts:
                continue
            embeddings = []
            batch_size = 32
            num_batches = (len(all_texts) + batch_size - 1) // batch_size
            expected_dimension = None
            for i in range(num_batches):
                batch_texts = all_texts[i * batch_size : (i + 1) * batch_size]
                batch_cache_file = cache_file.parent / f"{cache_file.stem}_batch_{i}{cache_file.suffix}"
                batch_embeddings = self.cache_manager.load_cached_data(batch_cache_file)
                if batch_embeddings is None:
                    if model_info["type"] == "hf":
                        batch_embeddings = self.compute_embeddings_hf(model, tokenizer, batch_texts)
                    elif model_info["type"] == "openai":
                        batch_embeddings = self.get_openai_embeddings_in_batches(batch_texts)
                    else:
                        batch_embeddings = np.array([])
                    self.cache_manager.cache_data(batch_embeddings, batch_cache_file)
                if batch_embeddings is not None and batch_embeddings.size > 0:
                    if expected_dimension is None:
                        expected_dimension = batch_embeddings.shape[1]
                    elif batch_embeddings.shape[1] != expected_dimension:
                        logging.error(
                            f"Embedding dimension mismatch in batch {i} for model '{model_name}' and category '{category_name}'. Expected dimension: {expected_dimension}, but got: {batch_embeddings.shape[1]}"
                        )
                        continue
                    embeddings.append(batch_embeddings)
                else:
                    logging.warning(f"No embeddings generated for batch {i} in category '{category_name}'.")
            if embeddings:
                embeddings = np.vstack(embeddings)
            else:
                embeddings = np.array([])
            embeddings_dict[category_name] = embeddings

            def compute_ids():
                return df["id"].tolist()

            self.cache_manager.get_or_compute(ids_file, compute_ids)

        if model_info["type"] == "hf":
            self.unload_model_and_tokenizer(model, tokenizer)
        return model_name, embeddings_dict

    def get_text_representation(self, row: pd.Series, text_fields: List[str]) -> str:
        fields = []
        for tfield in text_fields:
            value = row.get(tfield, "")
            if EmbeddingGenerator.is_missing_value(value):
                value = ""
            fields.append(value)
        text_values = []
        for field_value in fields:
            parsed_list = EmbeddingGenerator.parse_list_string(field_value)
            text_values.extend(parsed_list)
        text_values = [text for text in text_values if text.strip()]
        text_representation = " ".join(text_values).strip()
        return text_representation if text_representation else " "

    @staticmethod
    def is_missing_value(value: Any) -> bool:
        if isinstance(value, (list, np.ndarray, pd.Series)):
            return pd.isnull(value).all() or all(not str(v).strip() for v in value)
        else:
            return pd.isnull(value) or not str(value).strip()

    @staticmethod
    def parse_list_string(s: Any) -> List[str]:
        return Normalizer.parse_list_string(s)

    def process_models(
        self,
        model_names: List[str],
        positive_datasets: Dict[str, pd.DataFrame],
        negative_datasets: Dict[str, pd.DataFrame],
        seeds: Dict[str, int],
        text_fields: List[str],
        dataset_name: str = "default",
        cache_suffix: str = "",
    ) -> Dict[str, Dict[str, np.ndarray]]:
        embeddings_dict_all_models = {}
        for model_name in tqdm(model_names, desc="Processing models"):
            model_info = self.embedding_models_info.get(model_name)
            if not model_info:
                continue
            datasets_to_process = [
                ("positive", positive_datasets),
                ("negative", negative_datasets),
            ]
            for dataset_type, datasets in datasets_to_process:
                model_key = f"{model_name}_{dataset_type}"
                dataset_seed = seeds[dataset_type]
                _, embeddings_dict = self.process_model(
                    model_name=model_name,
                    model_info=model_info,
                    datasets=datasets,
                    seed=dataset_seed,
                    text_fields=text_fields,
                    dataset_name=dataset_name,
                    cache_suffix=cache_suffix,
                    dataset_type=dataset_type,
                )
                embeddings_dict_all_models[model_key] = embeddings_dict
        return embeddings_dict_all_models


class Pipeline:
    def __init__(
        self,
        config: Config,
        cache_manager: CacheManager,
        package_name: str,
        project_path: Path,
    ):
        self.config = config
        self.cache_manager = cache_manager
        self.package_name = package_name
        self.project_path = project_path
        self.catalog = None
        self.normalizer = Normalizer(cache_dir=cache_manager.cache_dir)
        self.data_loader = DataLoader(cache_manager=cache_manager, config=config, normalizer=self.normalizer)
        self.embedding_generator = EmbeddingGenerator(config=config, cache_manager=cache_manager)

    def _load_nodes(self) -> pd.DataFrame:
        try:
            configure_project(self.package_name)
            with KedroSession.create() as session:
                context = session.load_context()
                catalog = context.catalog
                self.catalog = catalog
                nodes_df = catalog.load(self.config.nodes_dataset_name)
                if isinstance(nodes_df, SparkDataFrame):
                    nodes_df = nodes_df.toPandas()
                return nodes_df
        except Exception:
            sys.exit(1)

    def load_data(
        self,
    ) -> Tuple[List[str], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame]:
        nodes_cache_file = self.cache_manager.cache_dir / "datasets" / f"nodes_df{self.config.cache_suffix}.pkl"
        nodes_df = self.cache_manager.get_or_compute(nodes_cache_file, self._load_nodes)

        edges_cache_file = self.cache_manager.cache_dir / "datasets" / f"edges_df{self.config.cache_suffix}.pkl"

        def load_edges():
            try:
                edges_df = self.catalog.load(self.config.edges_dataset_name)
                if isinstance(edges_df, SparkDataFrame):
                    edges_df = edges_df.toPandas()
                if edges_df.empty:
                    pass
                return edges_df
            except Exception:
                raise

        edges_df = self.cache_manager.get_or_compute(edges_cache_file, load_edges)

        categories, positive_datasets, negative_datasets, _ = self.data_loader.load_datasets(nodes_df=nodes_df)

        for dataset_type, datasets in [("positive", positive_datasets), ("negative", negative_datasets)]:
            for category, df in datasets.items():
                sampled_node_ids = set(df["id"].tolist())
                filtered_edges_df = edges_df[
                    edges_df["subject"].isin(sampled_node_ids) | edges_df["object"].isin(sampled_node_ids)
                ]
                edges_cache_file = (
                    self.cache_manager.cache_dir
                    / "datasets"
                    / f"edges_df_{dataset_type}_{category}{self.config.cache_suffix}.pkl"
                )
                filtered_edges_df.to_pickle(edges_cache_file)

        nodes_df.to_pickle(self.cache_manager.cache_dir / f"nodes_df{self.config.cache_suffix}.pkl")
        edges_df.to_pickle(self.cache_manager.cache_dir / f"edges_df{self.config.cache_suffix}.pkl")
        for category, df in positive_datasets.items():
            df.to_pickle(self.cache_manager.cache_dir / f"positive_df_{category}{self.config.cache_suffix}.pkl")
        for category, df in negative_datasets.items():
            df.to_pickle(self.cache_manager.cache_dir / f"negative_df_{category}{self.config.cache_suffix}.pkl")

        return categories, positive_datasets, negative_datasets, nodes_df


def safe_json_dumps(x: Any) -> str:
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, dict, set)):
        try:
            return json.dumps(x)
        except TypeError:
            return str(x)
    return json.dumps(x)


def main():
    Environment.configure_logging()
    utils_path = Path(__file__).parent.resolve()
    Environment.setup_environment(utils_path=utils_path)

    project_path = Path.cwd().parents[1]
    package_name = "matrix"

    cache_dir = project_path / "apps" / "embed_norm" / "cached_datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(cache_dir)
    for subdir in ["embeddings", "datasets"]:
        subdir_path = cache_dir / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)

    pos_seed = 54321
    neg_seed = 67890
    dataset_name = "rtx_kg2.int"
    nodes_dataset_name = "integration.int.rtx.nodes"
    edges_dataset_name = "integration.int.rtx.edges"
    categories = ["All Categories"]
    model_names = ["OpenAI", "PubMedBERT", "SapBERT", "BlueBERT", "BioBERT"]
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        sys.exit(1)
    openai.api_key = openai_api_key

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
    )

    cache_manager = CacheManager(cache_dir=cache_dir)

    pipeline = Pipeline(
        config=config,
        cache_manager=cache_manager,
        package_name=package_name,
        project_path=project_path,
    )

    categories, positive_datasets, negative_datasets, nodes_df = pipeline.load_data()

    seeds = {"positive": config.pos_seed, "negative": config.neg_seed}
    text_fields = ["name", "description"]
    pipeline.embedding_generator.process_models(
        model_names=config.model_names,
        positive_datasets=positive_datasets,
        negative_datasets=negative_datasets,
        seeds=seeds,
        text_fields=text_fields,
        dataset_name=config.dataset_name,
        cache_suffix=config.cache_suffix,
    )

    logging.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
