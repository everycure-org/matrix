# analyze.py

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, f1_score
from pathlib import Path
from tqdm.auto import tqdm
from multiprocessing import Pool
import multiprocessing
import gc
import pickle

from test import Config, CacheManager, setup_environment, configure_logging, set_up_variables


def load_embeddings(config, cache_manager, model_name: str, dataset_type: str, category: str):
    combinations_suffix = ""
    dataset_type_suffix = f"_{dataset_type}"
    cache_suffix = config.cache_suffix
    if dataset_type == "positive":
        seed = config.pos_seed
    else:
        seed = config.neg_seed
    cache_dir = cache_manager.cache_dir
    embeddings_file = (
        cache_dir
        / "embeddings"
        / f"{config.dataset_name}{dataset_type_suffix}_embeddings_{category}_{model_name}{combinations_suffix}_seed_{seed}{cache_suffix}.pkl"
    )
    ids_file = (
        cache_dir
        / "embeddings"
        / f"{config.dataset_name}{dataset_type_suffix}_ids_{category}_{model_name}{combinations_suffix}_seed_{seed}{cache_suffix}.pkl"
    )
    embeddings = cache_manager.load_cached_data(embeddings_file)
    ids = cache_manager.load_cached_data(ids_file)
    return embeddings, ids


def get_ground_truth_pairs(positive_df):
    id_to_indices = {}
    for idx, id_val in enumerate(positive_df["id"]):
        if id_val not in id_to_indices:
            id_to_indices[id_val] = []
        id_to_indices[id_val].append(idx)
    true_pairs_set = set()
    for indices in id_to_indices.values():
        if len(indices) > 1:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    true_pairs_set.add((indices[i], indices[j]))
    return true_pairs_set


def analyze_pair_worker(args):
    (model_name, category, config, cache_manager) = args
    pos_embeddings, pos_ids = load_embeddings(config, cache_manager, model_name, "positive", category)
    neg_embeddings, neg_ids = load_embeddings(config, cache_manager, model_name, "negative", category)
    if any(x is None for x in [pos_embeddings, neg_embeddings, pos_ids, neg_ids]):
        return model_name, category, None
    pos_embeddings = pos_embeddings.astype(np.float32, copy=False)
    neg_embeddings = neg_embeddings.astype(np.float32, copy=False)
    all_embeddings = np.vstack([pos_embeddings, neg_embeddings])
    all_ids = pos_ids + neg_ids
    n = len(all_embeddings)
    id_to_indices = {}
    for idx, id_val in enumerate(all_ids):
        id_to_indices.setdefault(id_val, []).append(idx)
    true_pairs_set = set()
    for indices in id_to_indices.values():
        if len(indices) > 1:
            true_pairs_set.update((i, j) for i in indices for j in indices if i < j)
    labels = []
    scores = []
    batch_size = 10000
    for i_start in range(0, n, batch_size):
        i_end = min(i_start + batch_size, n)
        embeddings_i = all_embeddings[i_start:i_end]
        indices_i = range(i_start, i_end)
        for j_start in range(i_start, n, batch_size):
            j_end = min(j_start + batch_size, n)
            embeddings_j = all_embeddings[j_start:j_end]
            indices_j = range(j_start, j_end)
            similarities = cosine_similarity(embeddings_i, embeddings_j)
            for idx_i, global_i in enumerate(indices_i):
                for idx_j, global_j in enumerate(indices_j):
                    if global_i >= global_j:
                        continue
                    score = similarities[idx_i, idx_j]
                    pair = (global_i, global_j)
                    is_true_pair = pair in true_pairs_set
                    labels.append(1 if is_true_pair else 0)
                    scores.append(score)
    del all_embeddings, pos_embeddings, neg_embeddings
    gc.collect()
    if len(set(labels)) < 2:
        return model_name, category, None
    roc_auc = roc_auc_score(labels, scores)
    avg_precision = average_precision_score(labels, scores)
    threshold = 0.975
    binary_preds = [1 if s >= threshold else 0 for s in scores]
    prec = precision_score(labels, binary_preds, zero_division=0)
    f1 = f1_score(labels, binary_preds, zero_division=0)
    total_pairs = len(labels)
    positive_pairs = sum(labels)
    negative_pairs = total_pairs - positive_pairs
    true_pos = sum((lab == 1 and pred == 1) for lab, pred in zip(labels, binary_preds))
    false_pos = sum((lab == 0 and pred == 1) for lab, pred in zip(labels, binary_preds))
    true_neg = sum((lab == 0 and pred == 0) for lab, pred in zip(labels, binary_preds))
    false_neg = sum((lab == 1 and pred == 0) for lab, pred in zip(labels, binary_preds))
    metrics = {
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "precision": prec,
        "f1": f1,
        "total_pairs": total_pairs,
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
        "true_positives": true_pos,
        "false_positives": false_pos,
        "true_negatives": true_neg,
        "false_negatives": false_neg,
        "correct_positive_predictions": true_pos,
        "correct_negative_predictions": true_neg,
        "incorrect_positive_predictions": false_pos,
        "incorrect_negative_predictions": false_neg,
    }
    return model_name, category, metrics


class EmbeddingAnalyzer:
    def __init__(self, config: Config, cache_manager: CacheManager):
        self.config = config
        self.cache_manager = cache_manager
        self.model_names = config.model_names
        self.metrics = {}
        self.categories = self.get_categories()

    def get_categories(self):
        categories_file = self.config.cache_dir / "datasets" / "rtx_kg2.int_categories.pkl"
        with open(categories_file, "rb") as f:
            categories = pickle.load(f)
        return categories

    def analyze(self):
        num_workers = max(1, multiprocessing.cpu_count() // 2)
        tasks = [
            (model_name, category, self.config, self.cache_manager)
            for model_name in self.model_names
            for category in self.categories
        ]
        with Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(analyze_pair_worker, tasks),
                    total=len(tasks),
                    desc="Analyzing",
                )
            )
        for model_name, category, metrics in results:
            if metrics:
                self.metrics.setdefault(model_name, {})[category] = metrics


def main():
    configure_logging()
    utils_path = Path(__file__).parent.resolve()
    setup_environment(utils_path=utils_path)
    root_path = Path.cwd().parents[1]
    config = set_up_variables(root_path=root_path)
    cache_manager = CacheManager(config.cache_dir)
    analyzer = EmbeddingAnalyzer(config, cache_manager)
    analyzer.analyze()
    metrics_output_dir = config.cache_dir / "analysis"
    metrics_output_dir.mkdir(parents=True, exist_ok=True)
    metrics_output_file = metrics_output_dir / "metrics.json"
    with open(metrics_output_file, "w") as f:
        json.dump(analyzer.metrics, f, indent=4)


if __name__ == "__main__":
    main()
