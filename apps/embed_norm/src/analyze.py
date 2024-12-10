import json
import numpy as np
import nmslib
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from tqdm.auto import tqdm
from multiprocessing import Pool
import multiprocessing
import pickle
from main import Config, CacheManager, Environment


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


def analyze_pair_with_sklearn(args):
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

    # Normalize embeddings
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    all_embeddings = all_embeddings / norms

    # Build the set of true pairs
    id_to_indices = {}
    for idx, id_val in enumerate(all_ids):
        id_to_indices.setdefault(id_val, []).append(idx)
    true_pairs_set = set()
    for indices in id_to_indices.values():
        if len(indices) > 1:
            true_pairs_set.update((i, j) for i in indices for j in indices if i < j)

    # Compute all pairwise cosine similarities
    print(f"Computing pairwise cosine similarities for {n} samples...")
    similarity_matrix = cosine_similarity(all_embeddings)

    # Get indices of upper triangle of the similarity matrix (excluding diagonal)
    upper_triangle_indices = np.triu_indices(n, k=1)
    pairs_i = upper_triangle_indices[0]
    pairs_j = upper_triangle_indices[1]
    scores = similarity_matrix[pairs_i, pairs_j]

    # Build labels for all pairs
    labels = []
    for i, j in zip(pairs_i, pairs_j):
        pair = (i, j) if i < j else (j, i)
        is_true_pair = 1 if pair in true_pairs_set else 0
        labels.append(is_true_pair)

    # Now compute the metrics
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
        f"predictions at threshold {str(threshold)}": {
            "true_positives": true_pos,
            "false_positives": false_pos,
            "true_negatives": true_neg,
            "false_negatives": false_neg,
        },
    }
    return model_name, category, metrics


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

    # Normalize embeddings
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    all_embeddings = all_embeddings / norms

    id_to_indices = {}
    for idx, id_val in enumerate(all_ids):
        id_to_indices.setdefault(id_val, []).append(idx)
    true_pairs_set = set()
    for indices in id_to_indices.values():
        if len(indices) > 1:
            true_pairs_set.update((i, j) for i in indices for j in indices if i < j)

    labels = []
    scores = []

    # Initialize NMSLIB index
    index = nmslib.init(method="hnsw", space="cosinesimil")
    # index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(all_embeddings)
    index.createIndex({"post": 2}, print_progress=False)  # Adjust 'post' parameter if needed

    k = 50
    neighbors = index.knnQueryBatch(all_embeddings, k=k + 1, num_threads=4)  # Adjust threads as needed

    for i in range(n):
        neighbor_ids, neighbor_distances = neighbors[i]
        for j_idx, j in enumerate(neighbor_ids[1:]):  # Skip the first neighbor (itself)
            if i < j:
                pair = (i, j)
            else:
                pair = (j, i)
            is_true_pair = 1 if pair in true_pairs_set else 0
            # Convert distance to similarity
            score = 1 - neighbor_distances[j_idx]
            labels.append(is_true_pair)
            scores.append(score)

    computed_pairs = set()
    for i in range(n):
        neighbor_ids, _ = neighbors[i]
        for j in neighbor_ids[1:]:
            pair = (min(i, j), max(i, j))
            computed_pairs.add(pair)

    remaining_true_pairs = true_pairs_set - computed_pairs
    if remaining_true_pairs:
        pairs_i = np.array([i for i, j in remaining_true_pairs])
        pairs_j = np.array([j for i, j in remaining_true_pairs])
        embeddings_i = all_embeddings[pairs_i]
        embeddings_j = all_embeddings[pairs_j]
        # Since embeddings are normalized, dot product gives cosine similarity
        similarities = np.sum(embeddings_i * embeddings_j, axis=1)
        labels.extend([1] * len(similarities))
        scores.extend(similarities.tolist())

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
        f"predictions at threshold {str(threshold)}": {
            "true_positives": true_pos,
            "false_positives": false_pos,
            "true_negatives": true_neg,
            "false_negatives": false_neg,
        },
    }
    return model_name, category, metrics


class EmbeddingAnalyzer:
    def __init__(self, config: Config, cache_manager: CacheManager):
        self.config = config
        self.cache_manager = cache_manager
        self.model_names = config.model_names
        self.metrics = {}
        self.categories = self.config.categories

    def analyze(self):
        num_workers = max(1, multiprocessing.cpu_count() // 2)
        tasks = [
            (model_name, category, self.config, self.cache_manager)
            for model_name in self.model_names
            for category in self.categories
        ]
        with Pool(processes=num_workers) as pool:
            # Run NMSLIB analysis
            results_nmslib = list(
                tqdm(
                    pool.imap_unordered(analyze_pair_worker, tasks),
                    total=len(tasks),
                    desc="Analyzing with NMSLIB",
                )
            )
        with Pool(processes=num_workers) as pool:
            # Run Sklearn analysis
            results_sklearn = list(
                tqdm(
                    pool.imap_unordered(analyze_pair_with_sklearn, tasks),
                    total=len(tasks),
                    desc="Analyzing with Sklearn",
                )
            )
        for (model_name_nmslib, category_nmslib, metrics_nmslib), (
            model_name_sklearn,
            category_sklearn,
            metrics_sklearn,
        ) in zip(results_nmslib, results_sklearn):
            if metrics_nmslib:
                self.metrics.setdefault(model_name_nmslib, {}).setdefault(category_nmslib, {})["nmslib"] = (
                    metrics_nmslib
                )
            if metrics_sklearn:
                self.metrics.setdefault(model_name_sklearn, {}).setdefault(category_sklearn, {})["sklearn"] = (
                    metrics_sklearn
                )


def main():
    Environment.configure_logging()
    utils_path = Path(__file__).parent.resolve()
    Environment.setup_environment(utils_path=utils_path)
    project_path = Path.cwd().parents[1]
    cache_dir = project_path / "apps" / "embed_norm" / "cached_datasets"
    pos_seed = 54321
    neg_seed = 67890
    dataset_name = "rtx_kg2.int"
    nodes_dataset_name = "integration.int.rtx.nodes"
    edges_dataset_name = "integration.int.rtx.edges"
    categories_file = f"{dataset_name}_categories.pkl"
    with open(cache_dir / "datasets" / categories_file, "rb") as f:
        categories = pickle.load(f)
    model_names = ["OpenAI", "PubMedBERT", "BioBERT", "BlueBERT", "SapBERT"]
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
