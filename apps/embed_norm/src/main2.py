#!/usr/bin/env python3
# main.py

import sys
import os
import logging
import subprocess
from pathlib import Path, PurePath, PurePosixPath
from typing import Any, Callable, Dict, List, Tuple, Union, Optional, Set
from dataclasses import dataclass, field, fields
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
from sentence_transformers import SentenceTransformer
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import rand
from kedro.framework.session import KedroSession
from kedro.framework.project import configure_project
from keybert import KeyBERT
import json
import ast
import tiktoken
import gc
from tqdm import tqdm

import requests
import time

PUBTATOR_BASE_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson?pmids="
PUBTATOR_BATCH_SIZE = 50
MAX_REQUESTS_PER_SECOND = 3


class KeyBERTExtractor:
    def __init__(self, keybert_model_names: List[str]):
        """
        Initializes KeyBERT models using SentenceTransformer on CPU.

        Args:
            keybert_model_names (List[str]): List of KeyBERT model names to initialize.
        """
        self.keybert_models = {}
        self.logger = logging.getLogger(__name__)

        for model_name in keybert_model_names:
            try:
                self.logger.info(f"Loading KeyBERT model '{model_name}' on CPU.")
                sentencemodel = SentenceTransformer(model_name, device="cpu")
                kw_model = KeyBERT(model=sentencemodel)
                self.keybert_models[model_name] = kw_model
                self.logger.info(f"Successfully loaded KeyBERT model '{model_name}' on CPU.")
            except Exception as e:
                self.logger.error(f"Failed to load KeyBERT model '{model_name}': {e}")

    def extract_key_phrases(self, text: str, model_name: str, top_n: int = 5) -> List[str]:
        """
        Extracts key phrases from the given text using the specified KeyBERT model.

        Args:
            text (str): The input text from which to extract key phrases.
            model_name (str): The name of the KeyBERT model to use.
            top_n (int, optional): Number of top key phrases to extract. Defaults to 5.

        Returns:
            List[str]: A list of extracted key phrases.
        """
        kw_model = self.keybert_models.get(model_name)
        if not kw_model:
            self.logger.error(f"KeyBERT model '{model_name}' not loaded.")
            return []

        try:
            phrases = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), top_n=top_n)
            return [phrase for phrase, _score in phrases]
        except Exception as e:
            self.logger.error(f"Error extracting key phrases with model '{model_name}': {e}")
            return []


class Environment:
    @staticmethod
    def setup_environment(utils_path: Path, root_subdir: str = "pipelines/matrix"):
        """
        Adjust sys.path, locate project root via Git, change directory, and load environment vars.
        """
        if str(utils_path) not in sys.path:
            sys.path.append(str(utils_path))
        try:
            root_path = Path(
                subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).decode().strip()
            )
            target_path = root_path / root_subdir
            os.chdir(target_path)
            load_dotenv(dotenv_path=target_path / ".env")
        except subprocess.CalledProcessError:
            sys.exit(1)
        except Exception:
            sys.exit(1)

    @staticmethod
    def configure_logging():
        """
        Basic global logging configuration.
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        logging.info("Logging is configured.")


class CacheManager:
    """
    Simple caching interface using joblib.
    """
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
        """
        Load from cache if exists; else compute, cache, then return.
        """
        cache_path = Path(cache_path)
        result = self.load_cached_data(cache_path)
        if result is not None:
            return result
        else:
            result = compute_func(*args, **kwargs)
            self.cache_data(result, cache_path)
            return result


def safe_json_dumps(x: Any) -> str:
    """
    Safely convert common Python containers to JSON strings.
    """
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, dict, set)):
        try:
            return json.dumps(x)
        except TypeError:
            return str(x)
    return json.dumps(x)


class Normalizer:
    """
    Asynchronously fetches normalized identifiers for given node IDs via Node Normalizer API.
    Caches results in 'normalized_data.pkl' to avoid repeated calls.
    """
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
        """
        POST to Node Normalizer with retries.
        """
        retries = 0
        while retries < self.MAX_RETRIES:
            try:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        retries += 1
                        await asyncio.sleep(2 ** retries)
            except aiohttp.ClientError:
                retries += 1
                await asyncio.sleep(2 ** retries)
        return None

    async def batch_normalize_curies_async(self, category_curies: Dict[str, List[str]]):
        """
        Sends batched node IDs to Normalizer.
        """
        all_curies = [curie for curies in category_curies.values() for curie in curies]
        curies_to_normalize = [curie for curie in all_curies if curie not in self.normalized_data]
        if not curies_to_normalize:
            return

        semaphore = asyncio.Semaphore(10)
        conn = aiohttp.TCPConnector(limit=10)

        async def sem_normalize(session, url, payload):
            async with semaphore:
                return await self.normalize_node(session, url, payload)

        async with aiohttp.ClientSession(connector=conn) as session:
            tasks = []
            for i in range(0, len(curies_to_normalize), self.BATCH_SIZE):
                batch = curies_to_normalize[i : i + self.BATCH_SIZE]
                payload = {"curies": batch, "conflate": False, "expand_all": True}
                task = asyncio.ensure_future(sem_normalize(session, self.NORM_URL, payload))
                tasks.append(task)

            responses = []
            for future in asyncio.as_completed(tasks):
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
        """
        Normalizes positive samples by calling Node Normalizer in batches.
        Returns an updated DataFrame with equivalent IDs expanded.
        """
        if positive_df.empty:
            return positive_df
        category_curies = {"positive": positive_df["id"].tolist()}
        asyncio.run(self.batch_normalize_curies_async(category_curies))
        equivalent_dfs = self.create_equivalent_items_dfs({"positive": positive_df}, self.normalized_data)
        return equivalent_dfs["positive"]

    @staticmethod
    def create_equivalent_items_dfs(
        positive_datasets: Dict[str, pd.DataFrame],
        normalized_data: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, pd.DataFrame]:
        """
        For each row in the dataframe, explode any 'equivalent_identifiers'
        and replace them with normalized info from Node Normalizer.
        """
        equivalent_dfs = {}
        for category, df in positive_datasets.items():
            if df.empty or "id" not in df.columns:
                equivalent_dfs[category] = df
                continue

            df["ids_to_normalize"] = df.apply(
                lambda row: list(
                    set(
                        [str(row["id"])]
                        + Normalizer.parse_list_string(row.get("equivalent_identifiers", []))
                    )
                ),
                axis=1,
            )
            df_exploded = df.explode("ids_to_normalize").reset_index(drop=True)
            df_exploded.rename(columns={"ids_to_normalize": "id_to_normalize"}, inplace=True)

            equivalents_list = []
            for row_item in df_exploded.to_dict(orient="records"):
                id_value = row_item["id_to_normalize"]
                if id_value in normalized_data:
                    equivalents = normalized_data[id_value]
                    for eq in equivalents:
                        new_row = row_item.copy()
                        new_row["id"] = eq["id"]
                        new_row["name"] = eq["name"]
                        new_row["all_categories"] = eq.get("types", [])
                        equivalents_list.append(new_row)
                else:
                    equivalents_list.append(row_item)

            equivalent_df = pd.DataFrame(equivalents_list)
            for col in equivalent_df.columns:
                if equivalent_df[col].apply(
                    lambda x: isinstance(x, (list, dict, set, np.ndarray))
                ).any():
                    equivalent_df[col] = equivalent_df[col].apply(safe_json_dumps)

            equivalent_df = equivalent_df.drop_duplicates()
            equivalent_dfs[category] = equivalent_df
        return equivalent_dfs

    @staticmethod
    def parse_list_string(s: Any) -> List[str]:
        """
        Safe conversion of string representation of list to an actual list.
        """
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
                return [s]
        else:
            return []


class DataLoader:
    """
    Loads data, obtains categories, samples positive/negative sets using Spark,
    then normalizes positives in pandas.
    Also provides convenience methods to retrieve 1-hop edges and sample them.
    """
    def __init__(self, cache_manager: CacheManager, config: "Config", normalizer: Normalizer):
        self.cache_manager = cache_manager
        self.config = config
        self.normalizer = normalizer
        # self.spark = SparkSession.builder.getOrCreate()

    def load_datasets(
        self, nodes_df: Union[SparkDataFrame, pd.DataFrame], edges_df: Union[SparkDataFrame, pd.DataFrame]
    ) -> Tuple[List[str], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
        """
        Convert nodes/edges to Spark if needed, then:
         1) Collect distinct categories in Spark
         2) Sample positives & negatives in Spark
         3) Collect them to Pandas
         4) Normalize the positive sets only
        Returns all as pandas DataFrames.
        """
        if isinstance(nodes_df, pd.DataFrame):
            nodes_sdf = self.spark.createDataFrame(nodes_df)
        else:
            nodes_sdf = nodes_df

        if isinstance(edges_df, pd.DataFrame):
            edges_sdf = self.spark.createDataFrame(edges_df)
        else:
            edges_sdf = edges_df

        categories = self.get_categories_spark(nodes_sdf)

        positive_datasets = {}
        negative_datasets = {}
        for category in tqdm(categories, desc="Spark sampling categories"):
            pos_sdf = self.sample_positive_spark(nodes_sdf, category)
            neg_sdf = self.sample_negative_spark(nodes_sdf, category)

            pos_pdf = pos_sdf.toPandas()
            neg_pdf = neg_sdf.toPandas()

            pos_pdf = self.normalizer.augment_positive_df(pos_pdf)

            positive_datasets[category] = pos_pdf
            negative_datasets[category] = neg_pdf

        nodes_pdf = nodes_sdf.toPandas()
        edges_pdf = edges_sdf.toPandas()

        return categories, positive_datasets, negative_datasets, nodes_pdf, edges_pdf

    def get_categories_spark(self, nodes_sdf: SparkDataFrame) -> List[str]:
        """
        Gather distinct categories from the Spark DataFrame; then add 'All Categories'.
        """
        distinct_categories = (
            nodes_sdf.select("category")
            .distinct()
            .dropna()
            .rdd.map(lambda row: row["category"])
            .collect()
        )
        cat_list = list(distinct_categories)
        cat_list.append("All Categories")
        return cat_list

    def sample_positive_spark(self, nodes_sdf: SparkDataFrame, category: str) -> SparkDataFrame:
        """
        Use Spark to sample the positive_n for a given category or entire set if 'All Categories'.
        """
        if category == "All Categories":
            subset_sdf = nodes_sdf
        else:
            subset_sdf = nodes_sdf.filter(F.col("category") == category)

        total_count = subset_sdf.count()
        actual_n = min(self.config.positive_n, total_count)
        if total_count == 0:
            return subset_sdf.limit(0)

        fraction = float(actual_n) / float(total_count) if total_count > 0 else 1.0
        sampled_sdf = subset_sdf.sample(withReplacement=False, fraction=fraction, seed=self.config.pos_seed)
        sampled_sdf = sampled_sdf.limit(actual_n)
        return sampled_sdf

    def sample_negative_spark(self, nodes_sdf: SparkDataFrame, category: str) -> SparkDataFrame:
        """
        Use Spark to sample the negative_n from nodes that are NOT the given category.
        If 'All Categories', sample from entire dataset.
        """
        if category == "All Categories":
            subset_sdf = nodes_sdf
        else:
            subset_sdf = nodes_sdf.filter(F.col("category") != category)

        total_count = subset_sdf.count()
        actual_n = min(self.config.negative_n, total_count)
        if total_count == 0:
            return subset_sdf.limit(0)

        fraction = float(actual_n) / float(total_count) if total_count > 0 else 1.0
        sampled_sdf = subset_sdf.sample(withReplacement=False, fraction=fraction, seed=self.config.neg_seed)
        sampled_sdf = sampled_sdf.limit(actual_n)
        return sampled_sdf


    def get_first_hop_info(
        self,
        edges_df: pd.DataFrame,
        nodes_df: pd.DataFrame,
        node_ids: Set[str],
        seed: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Retrieve edges that connect any of the given node_ids to other nodes (1-hop).
        Also return a DataFrame of the connected neighbor nodes.
        """
        first_hop_edges = edges_df[
            (edges_df["subject"].isin(node_ids)) | (edges_df["object"].isin(node_ids))
        ].copy()

        connected_node_ids = set(first_hop_edges["subject"].unique()) | set(first_hop_edges["object"].unique())
        connected_nodes_df = nodes_df[nodes_df["id"].isin(connected_node_ids)].copy()

        return first_hop_edges, connected_nodes_df

    def sample_edges_per_node(
        self,
        edges_df: pd.DataFrame,
        node_ids: List[str],
        sample_ratio: float,
        seed: int
    ) -> pd.DataFrame:
        """
        For each node in node_ids, retrieve all edges connecting that node,
        then sample a fraction of them (sample_ratio). Return the combined set.
        """
        np.random.seed(seed)

        all_sampled = []
        for node_id in node_ids:
            sub_edges = edges_df[(edges_df["subject"] == node_id) | (edges_df["object"] == node_id)]
            if not sub_edges.empty and 0.0 < sample_ratio < 1.0:
                sub_sampled = sub_edges.sample(frac=sample_ratio, random_state=seed)
                all_sampled.append(sub_sampled)
            else:
                if sample_ratio >= 1.0:
                    all_sampled.append(sub_edges)

        if all_sampled:
            final_df = pd.concat(all_sampled).drop_duplicates().reset_index(drop=True)
        else:
            final_df = pd.DataFrame(columns=edges_df.columns)
        return final_df


class EmbeddingGenerator:
    """
    Combines KeyBERT-based text expansions with either HF or OpenAI embeddings.
    """
    def __init__(self, config: "Config", cache_manager: CacheManager):
        self.config = config
        self.cache_manager = cache_manager
        self.embedding_models_info = config.embedding_models_info
        self.MAX_RETRIES = 3
        self.BATCH_SIZE = 100
        self.TOKENIZER_NAME = "text-embedding-3-small"
        self.TOKENIZER_ENCODING = "cl100k_base"

        self.keybert_extractor = KeyBERTExtractor(config.keybert_model_names)
        self.logger = logging.getLogger(__name__)


    @staticmethod
    def parse_pubtator_json(data: dict) -> dict:
        """
        Parse PubTator JSON => {pmid: combined_text}.
        """
        pmid_map = {}
        articles = data.get("articles", [])
        for article in articles:
            pmid = str(article.get("pmid", "")).strip()
            passages = article.get("passages", [])
            text_chunks = []
            for psg in passages:
                if "text" in psg:
                    txt = psg["text"]
                    if txt:
                        text_chunks.append(txt)
            combined = "\n".join(text_chunks)
            pmid_map[pmid] = combined
        return pmid_map

    @staticmethod
    def fetch_pubtator_chunk(pmids: list) -> dict:
        """
        Synchronous fetch of pmids from PubTator. Return pmid->abstract map.
        """
        if not pmids:
            return {}
        url = PUBTATOR_BASE_URL + ",".join(pmids)
        resp = requests.get(url)
        if resp.status_code != 200:
            logging.warning(f"PubTator returned status {resp.status_code} for PMIDs: {pmids}")
            return {}
        data = resp.json()
        return EmbeddingGenerator.parse_pubtator_json(data)

    @staticmethod
    def fetch_pubtator_data(all_pmids: list, batch_size: int = PUBTATOR_BATCH_SIZE) -> dict:
        """
        Collect all PMIDs in multiple chunks (synchronously),
        call fetch_pubtator_chunk, and merge the results.
        Enforces a 3 requests/second rate limit via time.sleep.
        """
        unique_pmids = list(set(all_pmids))
        final_map = {}

        for start in range(0, len(unique_pmids), batch_size):
            chunk = unique_pmids[start : start + batch_size]
            result = EmbeddingGenerator.fetch_pubtator_chunk(chunk)
            if isinstance(result, dict):
                final_map.update(result)
            else:
                logging.error(f"Error in PubTator batch for chunk {chunk}.")

            time.sleep(1.0 / MAX_REQUESTS_PER_SECOND + 0.05)

        return final_map

    @staticmethod
    def attach_pubtator_abstracts(df: pd.DataFrame, edges_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fully synchronous method to attach PubTator abstracts
        to each row in the DataFrame based on PMIDs found in edges_df.
        """
        if df.empty:
            df["pubtator_abstract"] = ""
            return df

        node_to_pmids = {}
        all_pmids = set()

        for node_id in df["id"].tolist():
            pmids_for_node = EmbeddingGenerator.get_pmids_for_row(node_id, edges_df)
            node_to_pmids[node_id] = pmids_for_node
            all_pmids.update(pmids_for_node)

        if not all_pmids:
            df["pubtator_abstract"] = ""
            return df

        pubtator_map = EmbeddingGenerator.fetch_pubtator_data(list(all_pmids), batch_size=PUBTATOR_BATCH_SIZE)

        abstracts_col = []
        for node_id in df["id"].tolist():
            pmids = node_to_pmids[node_id]
            if not pmids:
                abstracts_col.append("")
                continue
            text_pieces = []
            for pm in pmids:
                text_from_map = pubtator_map.get(pm, "")
                if text_from_map.strip():
                    text_pieces.append(text_from_map)
            abstracts_col.append("\n".join(text_pieces) if text_pieces else "")

        df["pubtator_abstract"] = abstracts_col
        return df

    @staticmethod
    def get_pmids_for_row(node_id: str, edges_df: pd.DataFrame) -> Set[str]:
        """
        Given a single node_id and edges_df,
        return a cleaned set of PMIDs from any edges referencing this node.
        """
        subset = edges_df[(edges_df["subject"] == node_id) | (edges_df["object"] == node_id)]
        pmids = set()

        for pubs in subset["publications"]:
            if pubs is None:
                continue
            if isinstance(pubs, float) and pd.isna(pubs):
                continue

            if isinstance(pubs, (list, set, np.ndarray)):
                for x in pubs:
                    if isinstance(x, str) and x.strip():
                        pmids.add(x.strip())
                continue

            if isinstance(pubs, str):
                if pubs.strip():
                    pmids.add(pubs.strip())
                continue

        cleaned = {p.replace("PMID:", "").strip() for p in pmids}
        final_pmids = {x for x in cleaned if x.isdigit()}
        return final_pmids


    def load_model_and_tokenizer(self, model_info: Dict[str, Any]) -> Tuple[AutoModel, AutoTokenizer]:
        """
        Load a Hugging Face model/tokenizer from 'model_info'.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_info["tokenizer_name"])
        model = AutoModel.from_pretrained(model_info["model_name"])
        model.to(device)
        model.eval()
        return model, tokenizer

    def unload_model_and_tokenizer(self, model: AutoModel, tokenizer: AutoTokenizer):
        """
        Safely remove model from memory, free GPU if used.
        """
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
        """
        Embedding for Hugging Face Transformers:
         - tokenizes text
         - runs forward pass
         - returns mean pooled hidden states
        """
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
        """
        Calls OpenAI Embedding API asynchronously. Retries on rate limit/connect errors.
        """
        if not texts or all(not text.strip() for text in texts):
            self.logger.warning("Empty or whitespace-only texts provided to OpenAI embeddings.")
            return np.array([])

        retries = 0
        while retries < self.MAX_RETRIES:
            try:
                async with AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")) as client:
                    response = await client.embeddings.create(input=texts, model=self.TOKENIZER_NAME)
                embeddings = [item["embedding"] for item in response.model_dump()["data"]]
                return np.array(embeddings)

            except openai.RateLimitError:
                retries += 1
                await asyncio.sleep(2 ** retries)
            except openai.APIStatusError:
                return None
            except openai.APIConnectionError:
                retries += 1
                await asyncio.sleep(2 ** retries)
            except Exception:
                retries += 1
                await asyncio.sleep(2 ** retries)

        return None

    async def get_openai_embeddings_in_batches_async(self, texts: List[str]) -> np.ndarray:
        """
        Break text list into sub-batches that respect token limits,
        request embeddings in multiple calls, and combine.
        """
        embeddings = []
        batches = self.batch_texts_by_token_limit(
            texts, 
            self.config.max_tokens_per_request,
            self.TOKENIZER_NAME
        )
        for batch in tqdm(batches, desc="Getting embeddings from OpenAI"):
            result = await self.get_openai_embedding_async(batch)
            if result is not None and result.size > 0:
                embeddings.append(result)
            else:
                self.logger.warning("No embeddings returned for a batch.")

        if embeddings:
            return np.vstack(embeddings)
        return np.array([])

    def get_openai_embeddings_in_batches(self, texts: List[str]) -> np.ndarray:
        """
        Synchronous wrapper around the async method.
        """
        return asyncio.run(self.get_openai_embeddings_in_batches_async(texts))

    @staticmethod
    def batch_texts_by_token_limit(
        texts: List[str],
        max_tokens_per_request: int = 8191,
        model_name: str = "text-embedding-3-small",
    ) -> List[List[str]]:
        """
        Splits `texts` into sub-batches so total token length in each batch 
        stays below `max_tokens_per_request`.
        """
        encoding = tiktoken.get_encoding("cl100k_base")
        batches = []
        current_batch = []
        current_tokens = 0

        for text in texts:
            tokens = encoding.encode(text)
            num_tokens = len(tokens)

            if num_tokens > max_tokens_per_request:
                tokens = tokens[:max_tokens_per_request]
                text = encoding.decode(tokens)
                num_tokens = max_tokens_per_request

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
        kb_model_name: str,
        model_info: Dict[str, Any],
        datasets: Dict[str, pd.DataFrame],
        edges_df: pd.DataFrame,
        nodes_df: pd.DataFrame,
        seed: int,
        text_fields: List[str],
        dataset_name: str = "default",
        cache_suffix: str = "",
        dataset_type: str = "negative",
    ) -> Tuple[str, Dict[str, np.ndarray]]:
        """
        Produces embeddings for each category's DataFrame in `datasets`.
        Each row is converted to text, then embedded by HF or OpenAI.
        Caches partial and final results.
        """
        embeddings_dict = {}
        combined_model_name = f"{model_name}_{kb_model_name}"

        if model_info["type"] == "hf":
            model, tokenizer = self.load_model_and_tokenizer(model_info)
        else:
            model = None
            tokenizer = None

        data_loader = DataLoader(
            cache_manager=self.cache_manager,
            config=self.config,
            normalizer=Normalizer(self.config.cache_dir)
        )

        for category_name, df in datasets.items():
            if df.empty:
                continue

            dataset_type_suffix = f"_{dataset_type}"
            cache_file = (
                self.cache_manager.cache_dir
                / "embeddings"
                / f"{dataset_name}{dataset_type_suffix}_embeddings_{category_name}_{combined_model_name}_seed_{seed}{cache_suffix}.pkl"
            )
            ids_file = (
                self.cache_manager.cache_dir
                / "embeddings"
                / f"{dataset_name}{dataset_type_suffix}_ids_{category_name}_{combined_model_name}_seed_{seed}{cache_suffix}.pkl"
            )

            embeddings = self.cache_manager.load_cached_data(cache_file)
            if embeddings is not None:
                embeddings_dict[category_name] = embeddings
                continue

            sampled_node_ids = set(df["id"].tolist())
            first_hop_edges, connected_nodes_df = data_loader.get_first_hop_info(
                edges_df, nodes_df, sampled_node_ids, seed
            )

            sampled_edges = data_loader.sample_edges_per_node(
                first_hop_edges, df["id"].tolist(), sample_ratio=0.6, seed=seed
            )

            all_texts = []
            for _, row in df.iterrows():
                text_repr = self.get_text_representation(
                    row,
                    text_fields,
                    sampled_edges,
                    connected_nodes_df,
                    sampled_node_ids,
                    kb_model_name
                )
                all_texts.append(text_repr)

            if not all_texts:
                embeddings_dict[category_name] = np.array([])
                self.cache_manager.cache_data(np.array([]), cache_file)
                continue

            final_embeddings = []
            batch_size = 32
            num_batches = (len(all_texts) + batch_size - 1) // batch_size
            expected_dim = None

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
                    if expected_dim is None:
                        expected_dim = batch_embeddings.shape[1]
                    elif batch_embeddings.shape[1] != expected_dim:
                        self.logger.error(
                            f"Embedding dimension mismatch in batch {i} for model "
                            f"'{model_name}' (category '{category_name}'). "
                            f"Expected dimension: {expected_dim}, got: {batch_embeddings.shape[1]}"
                        )
                        continue
                    final_embeddings.append(batch_embeddings)
                else:
                    self.logger.warning(f"No embeddings generated for batch {i} (category '{category_name}').")

            if final_embeddings:
                final_embeddings = np.vstack(final_embeddings)
            else:
                final_embeddings = np.array([])

            embeddings_dict[category_name] = final_embeddings
            self.cache_manager.cache_data(final_embeddings, cache_file)

            def compute_ids():
                return df["id"].tolist()

            _id_list = self.cache_manager.get_or_compute(ids_file, compute_ids)

        if model_info["type"] == "hf":
            self.unload_model_and_tokenizer(model, tokenizer)

        return combined_model_name, embeddings_dict

    def get_text_representation(
        self,
        row: pd.Series,
        text_fields: List[str],
        sampled_edges_df: pd.DataFrame,
        connected_nodes_df: pd.DataFrame,
        sampled_node_ids: Set[str],
        kb_model_name: str,
    ) -> str:
        """
        Build textual context for a node, including:
          1) Node info (category, name, description)
          2) Key edges & neighbors
          3) Key phrases from PubTator abstracts (if any)
        """
        subject_node = {
            "categories": self.parse_list_string(row.get("category", [])),
            "labels": self.parse_list_string(row.get("name", [])),
        }
        node_description = row.get("description", "")
        node_id = row["id"]

        text_parts = []
        text_parts.append("=== NODE INFORMATION ===")
        if subject_node["labels"]:
            text_parts.append(f"Name: {', '.join(subject_node['labels'])}")
        if subject_node["categories"]:
            text_parts.append(f"Category: {', '.join(subject_node['categories'])}")
        if node_description:
            text_parts.append(f"Description: {node_description}")

        node_edges = sampled_edges_df[
            (sampled_edges_df["subject"] == node_id) | (sampled_edges_df["object"] == node_id)
        ]
        if not node_edges.empty:
            text_parts.append("=== CONNECTIONS ===")

        for _, edge_row in node_edges.iterrows():
            predicate = edge_row["predicate"]
            neighbor_id = (
                edge_row["object"] if edge_row["subject"] == node_id else edge_row["subject"]
            )
            if neighbor_id in sampled_node_ids:
                continue

            neighbor_row = connected_nodes_df[connected_nodes_df["id"] == neighbor_id]
            if neighbor_row.empty:
                continue

            neighbor_name = neighbor_row.iloc[0].get("name", "")
            neighbor_cats = self.parse_list_string(neighbor_row.iloc[0].get("category", []))
            text_parts.append(
                f"Connected to {neighbor_name} (categories: {', '.join(neighbor_cats)}) via {predicate}."
            )

        pmids = set(self.parse_list_string(row.get("pmid", [])))
        for _, edge_row in node_edges.iterrows():
            pmids.update(self.parse_list_string(edge_row.get("publications", [])))

        cleaned_pmids = {self.clean_pmid(pmid) for pmid in pmids if pmid}
        cleaned_pmids = {pmid for pmid in cleaned_pmids if pmid.isdigit()}

        pubtator_map = self.fetch_pubtator_data(list(cleaned_pmids))

        if pubtator_map:
            text_parts.append("=== KEY PHRASES FROM ABSTRACTS ===")
            key_phrases_all = []
            for pmid_val, abstract in pubtator_map.items():
                if abstract.strip():
                    phrases = self.keybert_extractor.extract_key_phrases(abstract, kb_model_name, top_n=8)
                    key_phrases_all.extend(phrases)

            seen_phrases = set()
            key_phrases_unique = []
            for phrase in key_phrases_all:
                if phrase not in seen_phrases:
                    key_phrases_unique.append(phrase)
                    seen_phrases.add(phrase)

            if key_phrases_unique:
                phrases_str = ", ".join(key_phrases_unique)
                text_parts.append(f"Key Phrases: {phrases_str}")

        full_text = "\n".join(text_parts)
        final_text = self.chunk_text_if_needed(full_text, max_tokens=1024)
        return final_text

    @staticmethod
    def chunk_text_if_needed(text: str, max_tokens: int = 1024) -> str:
        """
        If the text is too long, chunk it by words,
        joining chunks with '=== CHUNK BREAK ==='.
        """
        words = text.split()
        if len(words) <= max_tokens:
            return text

        chunks = []
        start = 0
        while start < len(words):
            end = min(start + max_tokens, len(words))
            chunk = words[start:end]
            chunk_str = " ".join(chunk)
            chunks.append(chunk_str)
            start += max_tokens
        return "\n=== CHUNK BREAK ===\n".join(chunks)

    @staticmethod
    def clean_pmid(pmid: str) -> str:
        return pmid.replace("PMID:", "").strip()

    @staticmethod
    def is_missing_value(value: Any) -> bool:
        """
        Detect if a cell is empty/NaN.
        """
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
        edges_df: pd.DataFrame,
        nodes_df: pd.DataFrame,
        seeds: Dict[str, int],
        text_fields: List[str],
        dataset_name: str = "default",
        cache_suffix: str = "",
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Loop over all embedding model names & KeyBERT models,
        producing embeddings for both the positive and negative sets.
        Returns a dict of { "<model>_<kbmodel>_<pos/neg>": {category: np.ndarray} }
        """
        embeddings_dict_all_models = {}

        for model_name in tqdm(model_names, desc="Processing models"):
            model_info = self.embedding_models_info.get(model_name)
            if not model_info:
                continue

            datasets_to_process = [
                ("positive", positive_datasets),
                ("negative", negative_datasets),
            ]

            for kb_model_name in self.keybert_extractor.keybert_models.keys():
                for dataset_type, datasets in datasets_to_process:
                    dataset_seed = seeds[dataset_type]
                    combined_model_name, embeddings_dict = self.process_model(
                        model_name=model_name,
                        kb_model_name=kb_model_name,
                        model_info=model_info,
                        datasets=datasets,
                        edges_df=edges_df,
                        nodes_df=nodes_df,
                        seed=dataset_seed,
                        text_fields=text_fields,
                        dataset_name=dataset_name,
                        cache_suffix=cache_suffix,
                        dataset_type=dataset_type,
                    )
                    model_key = f"{combined_model_name}_{dataset_type}"
                    embeddings_dict_all_models[model_key] = embeddings_dict

        return embeddings_dict_all_models


@dataclass
class Config:
    """
    Holds global settings, including environment, dataset names,
    model info, sampling ratio, logging configuration, etc.
    """
    project_path: Path = Path.cwd().parents[0]
    pos_seed: int = 54321
    neg_seed: int = 67890
    dataset_name: str = "rtx_kg2.int"
    nodes_dataset_name: str = "integration.int.rtx_kg2.nodes"
    edges_dataset_name: str = "integration.int.rtx_kg2.edges"
    categories: List[str] = field(default_factory=lambda: [])
    total_sample_size: int = 32
    positive_ratio: float = 0.2
    num_folds: int = 5
    package_name: str = "matrix"

    embedding_models_info: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "OpenAI": {
                "type": "openai",
                "model_name": "text-embedding-3-small",
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
            "BioSynSAPBERT-ncbi-disease": {
                "type": "hf",
                "tokenizer_name": "dmis-lab/biosyn-sapbert-ncbi-disease",
                "model_name": "dmis-lab/biosyn-sapbert-ncbi-disease",
            },
            "BioSynSAPBERT-disease": {
                "type": "hf",
                "tokenizer_name": "dmis-lab/biosyn-sapbert-bc5cdr-disease",
                "model_name": "dmis-lab/biosyn-sapbert-bc5cdr-disease",
            },
            "BioSynSAPBERT-chemical": {
                "type": "hf",
                "tokenizer_name": "dmis-lab/biosyn-sapbert-bc5cdr-chemical",
                "model_name": "dmis-lab/biosyn-sapbert-bc5cdr-chemical",
            },
            "BioSynSAPBERT": {
                "type": "hf",
                "tokenizer_name": "dmis-lab/biosyn-sapbert-bc2gn",
                "model_name": "dmis-lab/biosyn-sapbert-bc2gn",
            },
            "DiLBERT-DiseaseLanguage": {
                "type": "hf",
                "tokenizer_name": "beatrice-portelli/DiLBERT",
                "model_name": "beatrice-portelli/DiLBERT",
            },
        }
    )
    keybert_model_names: List[str] = field(
        default_factory=lambda: [
            "sentence-transformers/all-MiniLM-L6-v2",
            "distilbert-base-nli-stsb-mean-tokens",
            "roberta-base-nli-stsb-mean-tokens",
            "sentence-transformers/allenai-specter",
            "FremyCompany/BioLORD-2023",
            "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
            "FremyCompany/BioLORD-2023-C",
            "FremyCompany/BioLORD-2023-M",
            "microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL",
            "microsoft/BiomedNLP-BiomedELECTRA-large-uncased-abstract",
            "cambridgeltl/BioRedditBERT-uncased",
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
            "dmis-lab/biosyn-sapbert-bc2gn",
            "beatrice-portelli/DiLBERT",
            "dmis-lab/biosyn-sapbert-bc5cdr-chemical",
            "dmis-lab/biosyn-sapbert-ncbi-disease",
            "dmis-lab/biosyn-sapbert-bc5cdr-disease",
        ]
    )

    api_services_info: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "PubTator": {
                "base_url": "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson?pmids=",
                "max_calls": 3,
                "period": 1.1,
            },
            "OpenAI": {
                "base_url": "https://api.openai.com/v1",
                "max_calls": 20,
                "period": 60
            },
            "Normalizer": {
                "base_url": "https://nodenorm.transltr.io/1.5",
                "max_calls": 10,
                "period": 1
            },
        }
    )

    positive_n: int = field(init=False)
    negative_n: int = field(init=False)
    cache_suffix: str = field(init=False)
    model_names: List[str] = field(init=False)
    utils_path: Optional[Path] = field(init=False)
    cache_dir: Optional[Path] = field(init=False)
    profiling_dir: Optional[Path] = field(init=False)
    pipeline_log_file_path: Optional[Path] = field(init=False)
    openai_api_key: Optional[str] = field(init=False)
    embedding_log_file_path: Optional[Path] = None

    logger: logging.Logger = field(init=False, default=None)

    max_tokens_per_request: int = 8191

    def __post_init__(self):
        self.setup_logger()

        self.positive_n: int = int(self.total_sample_size * self.positive_ratio)
        self.negative_n: int = self.total_sample_size - self.positive_n
        self.cache_suffix: str = f"_pos_{self.positive_n}_neg_{self.negative_n}"
        self.utils_path = self.project_path / "src"
        self.cache_dir = self.project_path / "cached_datasets"
        self.profiling_dir = self.cache_dir / "profiling"
        self.pipeline_log_file_path = self.profiling_dir / "embed_pipeline.log"
        self.embedding_log_file_path = self.profiling_dir / "embedding_generator.log"
        self.paths = {
            "cache_dir": self.cache_dir,
            "profiling_dir": self.profiling_dir,
            "utils_path": self.utils_path,
            "project_path": self.project_path,
        }
        self.model_names = list(self.embedding_models_info.keys())
        Environment.setup_environment(utils_path=self.utils_path)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if not self.openai_api_key:
            self.logger.error("OPENAI_API_KEY not found in environment. Exiting.")
            sys.exit(1)

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a dict representation, excluding openai_api_key for security.
        """
        state = {}
        for f in fields(self):
            if f.name != "openai_api_key":
                state[f.name] = getattr(self, f.name)
        return state

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """
        Instantiate from a dict, ignoring or setting non-init fields.
        """
        init_fields = {f.name for f in fields(cls) if f.init}
        init_args = {k: v for k, v in config_dict.items() if k in init_fields}
        other_fields = {k: v for k, v in config_dict.items() if k not in init_fields}
        config = cls(**init_args)
        for field_name, field_value in other_fields.items():
            if field_name != "openai_api_key":
                setattr(config, field_name, field_value)
        config.__post_init__()
        config.openai_api_key = os.getenv("OPENAI_API_KEY")
        return config

    def setup_logger(self):
        """
        Initialize a file + stream logger for the config object, if not already.
        """
        if self.logger is None:
            self.logger = logging.getLogger("ConfigLogger")
            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [Config] %(message)s")

            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

            if hasattr(self, "pipeline_log_file_path") and self.pipeline_log_file_path:
                self.pipeline_log_file_path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(self.pipeline_log_file_path)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
                self.file_handler = file_handler

    def __getstate__(self):
        """
        Exclude non-picklable attributes (like logger) from the pickled state.
        """
        state = self.__dict__.copy()
        if "logger" in state:
            del state["logger"]
        if "file_handler" in state:
            del state["file_handler"]
        return state

    def __setstate__(self, state):
        """
        Restore attributes after unpickling, re-initialize logger, etc.
        """
        self.__dict__.update(state)
        self.setup_logger()


class Pipeline:
    """
    Orchestrates data loading, sampling, normalization, and embedding generation.
    """
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
        self.project_path = config.project_path
        root_path = Path(__file__).resolve().parents[3]
        target_path = root_path / "pipelines/matrix"
        os.chdir(target_path)
        dotenv_path = target_path / ".env"
        if dotenv_path.is_file():
            load_dotenv(dotenv_path=dotenv_path)
        
        self.catalog = None
        self.normalizer = Normalizer(cache_dir=cache_manager.cache_dir)
        self.data_loader = DataLoader(
            cache_manager=cache_manager,
            config=config,
            normalizer=self.normalizer
        )
        self.embedding_generator = EmbeddingGenerator(
            config=config,
            cache_manager=cache_manager
        )
        




    def _load_nodes_edges_spark(self) -> Tuple[SparkDataFrame, SparkDataFrame]:
        """
        Load nodes/edges from Kedro as Spark DataFrames, if available.
        """
        try:
            configure_project(self.package_name)
            with KedroSession.create(env="cloud") as session:
                context = session.load_context()
                catalog = context.catalog
                self.catalog = catalog
                catalog.list()
                nodes_sdf = catalog.load(self.config.nodes_dataset_name)
                edges_sdf = catalog.load(self.config.edges_dataset_name)
                if not isinstance(nodes_sdf, SparkDataFrame) or not isinstance(edges_sdf, SparkDataFrame):
                    raise ValueError(
                        "Expected SparkDataFrame but got something else. "
                        "Check your Kedro dataset configuration."
                    )
                return nodes_sdf, edges_sdf
        except Exception as e:
            logging.error(f"Error loading nodes/edges via Kedro (Spark): {e}")
            sys.exit(1)

    def load_data(
        self,
    ) -> Tuple[List[str], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
        """
        Loads or retrieves from cache:
          - Spark DataFrames for nodes_df, edges_df
          - Then calls DataLoader to produce category splits, positive/negative sets
          - Collects them to Pandas, normalizes positives, returns them
        """
        nodes_edges_cache_file = (
            self.cache_manager.cache_dir / "datasets" / f"spark_nodes_edges_df{self.config.cache_suffix}.pkl"
        )

        def load_nodes_edges_spark():
            return "Spark nodes/edges loaded"

        _ = self.cache_manager.get_or_compute(nodes_edges_cache_file, load_nodes_edges_spark)

        nodes_sdf, edges_sdf = self._load_nodes_edges_spark()

        categories, positive_datasets, negative_datasets, nodes_pdf, edges_pdf = self.data_loader.load_datasets(
            nodes_sdf, edges_sdf
        )

        for dataset_type, datasets in [("positive", positive_datasets), ("negative", negative_datasets)]:
            for category, df in datasets.items():
                sampled_node_ids = set(df["id"].tolist())
                filtered_edges_df = edges_pdf[
                    edges_pdf["subject"].isin(sampled_node_ids) | edges_pdf["object"].isin(sampled_node_ids)
                ]
                edges_cache_file = (
                    self.cache_manager.cache_dir
                    / "datasets"
                    / f"edges_df_{dataset_type}_{category}{self.config.cache_suffix}.pkl"
                )
                filtered_edges_df.to_pickle(edges_cache_file)

        nodes_pdf.to_pickle(self.cache_manager.cache_dir / "datasets" / f"nodes_df{self.config.cache_suffix}.pkl")
        edges_pdf.to_pickle(self.cache_manager.cache_dir / "datasets" / f"edges_df{self.config.cache_suffix}.pkl")

        for category, df in positive_datasets.items():
            df.to_pickle(
                self.cache_manager.cache_dir / "datasets" / f"positive_df_{category}{self.config.cache_suffix}.pkl"
            )
        for category, df in negative_datasets.items():
            df.to_pickle(
                self.cache_manager.cache_dir / "datasets" / f"negative_df_{category}{self.config.cache_suffix}.pkl"
            )

        return categories, positive_datasets, negative_datasets, nodes_pdf, edges_pdf



def main():
    Environment.configure_logging()

    utils_path = Path(__file__).parent.resolve()
    Environment.setup_environment(utils_path=utils_path)

    project_path = Path.cwd().resolve()
    package_name = "matrix"

    cache_dir = project_path.parent / "cached_datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["embeddings", "datasets"]:
        subdir_path = cache_dir / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logging.error("OPENAI_API_KEY is missing. Exiting.")
        sys.exit(1)
    openai.api_key = openai_api_key

    config = Config()
    cache_manager = CacheManager(cache_dir=cache_dir)
    pipeline = Pipeline(
        config=config,
        cache_manager=cache_manager,
        package_name=package_name,
        project_path=project_path,
    )

    categories, positive_datasets, negative_datasets, nodes_df, edges_df = pipeline.load_data()

    seeds = {"positive": config.pos_seed, "negative": config.neg_seed}

    text_fields = ["name", "description"]

    pipeline.embedding_generator.process_models(
        model_names=config.model_names,
        positive_datasets=positive_datasets,
        negative_datasets=negative_datasets,
        edges_df=edges_df,
        nodes_df=nodes_df,
        seeds=seeds,
        text_fields=text_fields,
        dataset_name=config.dataset_name,
        cache_suffix=config.cache_suffix,
    )

    logging.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
