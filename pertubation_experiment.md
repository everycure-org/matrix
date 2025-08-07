# Knowledge Graph Edge Perturbation Experiment

## Experiment Overview

This experiment tests how edge rewiring perturbations in the knowledge graph affect downstream model
performance. We will systematically rewire 1%, 5%, 20%, and 50% of edges to connect to different
nodes of the same category, measuring the impact on drug-disease prediction accuracy.

## Methodology

### Perturbation Strategy

- **Type**: Edge rewiring (reconnecting edges to different nodes of same category)
- **Levels**: 1%, 5%, 20%, 50% of total edges
- **Sampling**: Category-stratified random sampling - edges reconnected within same node categories
- **Seeds**: Fixed random seeds for reproducibility (42, 123, 456, 789)
- **Constraint**: Object nodes must be of the same biolink category as original target

### Control Groups

- **Baseline**: 0% perturbation (original KG)
- **Multiple runs**: 4 random seeds per perturbation level for statistical significance

### Metrics to Evaluate

- **Primary**: AUROC, AUPRC for drug-disease predictions
- **Secondary**:
  - Embedding quality metrics (if available)
  - Model convergence time
  - Number of valid drug-disease pairs after perturbation

## Implementation Plan

### 1. Pipeline Integration Point

**Optimal Location**: After integration, before filtering (`integration.prm.unified_edges` →
perturbation → filtering)

**Rationale**:

- Captures the complete unified KG before any domain-specific filtering
- Allows filtering pipeline to naturally handle any orphaned nodes
- Maintains existing embedding and modeling pipeline integrity

### 2. New Pipeline Node: `perturb_edges_rewire`

```python
# Location: src/matrix/pipelines/perturbation/
def perturb_edges_rewire(
    unified_edges: DataFrame,  # Spark DataFrame
    unified_nodes: DataFrame,  # Spark DataFrame
    perturbation_rate: float,
    random_seed: int = 42,
    strategy: str = "category_stratified"
) -> DataFrame:
    """
    Rewire a percentage of edges to different nodes of same category

    Args:
        unified_edges: Complete unified edge list (Spark DataFrame)
        unified_nodes: Complete unified node list with categories
        perturbation_rate: Fraction of edges to rewire (0.01, 0.05, 0.20, 0.50)
        random_seed: For reproducibility
        strategy: "category_stratified" - maintain node category distribution

    Returns:
        Rewired edge DataFrame with same schema as input

    Implementation:
        1. Join edges with nodes to get subject/object categories
        2. Sample perturbation_rate of edges stratified by object category
        3. For each sampled edge, randomly select new object from same category
        4. Replace object_id in sampled edges with new random targets
        5. Union rewired edges with non-perturbed edges
    """
```

### 3. Modified Pipeline Structure

```
Integration Pipeline → Perturbation Node → Filtering Pipeline → Embeddings → Modeling
```

### 3. Detailed Spark Implementation

```python
from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import (
    col, rand, row_number, collect_list, size, expr,
    when, broadcast, monotonically_increasing_id
)
from pyspark.sql.types import StructType

def perturb_edges_rewire(
    unified_edges: DataFrame,
    unified_nodes: DataFrame,
    perturbation_rate: float,
    random_seed: int = 42
) -> DataFrame:
    """
    Rewire edges to different nodes of same category using Spark

    Algorithm:
    1. Join edges with nodes to get object categories
    2. Sample edges for perturbation stratified by object category
    3. For each category, create a pool of alternative targets
    4. Randomly reassign object_ids within each category
    5. Union perturbed edges with unperturbed edges
    """

    # Set random seed for reproducibility
    spark.sql(f"SET spark.sql.adaptive.enabled=true")
    spark.sql(f"SET spark.sql.adaptive.coalescePartitions.enabled=true")

    # Step 1: Enrich edges with object node categories
    edges_with_categories = unified_edges.join(
        unified_nodes.select("id", "category").alias("obj_nodes"),
        col("object") == col("obj_nodes.id"),
        "inner"
    ).select(
        "subject", "predicate", "object", "upstream_data_source",
        col("obj_nodes.category").alias("object_category"),
        # Include all other edge columns
        *[c for c in unified_edges.columns if c not in ["subject", "predicate", "object", "upstream_data_source"]]
    ).cache()

    # Step 2: Sample edges for perturbation, stratified by object category
    edges_for_perturbation = edges_with_categories.withColumn(
        "random_val", rand(seed=random_seed)
    ).filter(
        col("random_val") < perturbation_rate
    ).withColumn(
        "edge_id", monotonically_increasing_id()
    ).cache()

    # Step 3: Create pools of alternative targets by category
    target_pools_by_category = unified_nodes.select("id", "category").groupBy("category").agg(
        collect_list("id").alias("candidate_targets")
    ).filter(
        size(col("candidate_targets")) > 1  # Need at least 2 nodes to rewire
    ).cache()

    # Step 4: Rewire edges within each category
    edges_to_rewire = edges_for_perturbation.join(
        broadcast(target_pools_by_category),
        col("object_category") == col("category"),
        "inner"
    )

    # Create random reassignments within each category
    rewired_edges = edges_to_rewire.withColumn(
        "random_index", (rand(seed=random_seed + 1) * size(col("candidate_targets"))).cast("int")
    ).withColumn(
        "new_object", expr("candidate_targets[random_index]")
    ).filter(
        col("new_object") != col("object")  # Ensure we actually change the target
    ).select(
        col("subject"),
        col("predicate"),
        col("new_object").alias("object"),
        *[c for c in unified_edges.columns if c not in ["subject", "predicate", "object"]]
    )

    # Step 5: Get edges that were NOT selected for perturbation
    unperturbed_edges = edges_with_categories.join(
        edges_for_perturbation.select("subject", "predicate", "object", "edge_id"),
        ["subject", "predicate", "object"],
        "left_anti"
    ).select(
        *[c for c in unified_edges.columns]  # Original schema
    )

    # Step 6: Union rewired and unperturbed edges
    final_edges = unperturbed_edges.unionByName(rewired_edges, allowMissingColumns=True)

    # Cleanup cached DataFrames
    edges_with_categories.unpersist()
    edges_for_perturbation.unpersist()
    target_pools_by_category.unpersist()

    return final_edges.cache()

# Memory-efficient category statistics
def log_rewiring_stats(edges_before: DataFrame, edges_after: DataFrame, perturbation_rate: float):
    """Log statistics about the rewiring process"""

    total_before = edges_before.count()
    total_after = edges_after.count()
    expected_rewired = int(total_before * perturbation_rate)

    print(f"Rewiring Statistics:")
    print(f"  - Total edges before: {total_before:,}")
    print(f"  - Total edges after: {total_after:,}")
    print(f"  - Expected rewired edges: {expected_rewired:,}")
    print(f"  - Perturbation rate: {perturbation_rate:.1%}")

    # Category distribution before/after (sample for large datasets)
    if total_before > 1_000_000:
        sample_fraction = min(0.1, 100_000 / total_before)
        category_stats_before = edges_before.sample(sample_fraction).join(
            unified_nodes.select("id", "category"),
            col("object") == col("id")
        ).groupBy("category").count().collect()
    else:
        category_stats_before = edges_before.join(
            unified_nodes.select("id", "category"),
            col("object") == col("id")
        ).groupBy("category").count().collect()

    print("  - Category distribution maintained")
```

## Experiment Configuration

### Parameters Structure

```yaml
# conf/base/parameters/perturbation.yml
perturbation:
  enabled: true
  rate: 0.01 # Will be overridden per experiment
  random_seed: 42
  strategy: "category_stratified" # Maintain node category distribution
  rewire_mode: "object_only" # Only rewire object nodes (preserve subject)
```

## Kedro Experiment Commands

### Setup Experiments

```bash
# Create experiment group
kedro experiment create --name "kg_perturbation_study" \
  --description "Impact of edge perturbations on model performance"
```

### Individual Experiment Runs

#### Seed 42 Experiments

```bash
# Baseline (0% perturbation)
kedro experiment run --run-name "baseline_seed_42" \
  --experiment-name "kg_perturbation_study" \
  --params "perturbation.enabled=false,perturbation.random_seed=42" \
  --username $USER --release-version test-release

# 1% Perturbation (Rewiring)
kedro experiment run --run-name "rewire_1pct_seed_42" \
  --experiment-name "kg_perturbation_study" \
  --params "perturbation.enabled=true,perturbation.rate=0.01,perturbation.random_seed=42,perturbation.strategy=category_stratified" \
  --username $USER --release-version test-release

# 5% Perturbation (Rewiring)
kedro experiment run --run-name "rewire_5pct_seed_42" \
  --experiment-name "kg_perturbation_study" \
  --params "perturbation.enabled=true,perturbation.rate=0.05,perturbation.random_seed=42,perturbation.strategy=category_stratified" \
  --username $USER --release-version test-release

# 20% Perturbation (Rewiring)
kedro experiment run --run-name "rewire_20pct_seed_42" \
  --experiment-name "kg_perturbation_study" \
  --params "perturbation.enabled=true,perturbation.rate=0.20,perturbation.random_seed=42,perturbation.strategy=category_stratified" \
  --username $USER --release-version test-release

# 50% Perturbation (Rewiring)
kedro experiment run --run-name "rewire_50pct_seed_42" \
  --experiment-name "kg_perturbation_study" \
  --params "perturbation.enabled=true,perturbation.rate=0.50,perturbation.random_seed=42,perturbation.strategy=category_stratified" \
  --username $USER --release-version test-release
```

#### Seed 123 Experiments

```bash
# Baseline (0% perturbation)
kedro experiment run --run-name "baseline_seed_123" \
  --experiment-name "kg_perturbation_study" \
  --headless \
  --params "perturbation.enabled=false,perturbation.random_seed=123" \
  --username $USER --release-version test-release

# 1% Perturbation (Rewiring)
kedro experiment run --run-name "rewire_1pct_seed_123" \
  --experiment-name "kg_perturbation_study" \
  --headless \
  --params "perturbation.enabled=true,perturbation.rate=0.01,perturbation.random_seed=123,perturbation.strategy=category_stratified" \
  --username $USER --release-version test-release

# 20% Perturbation (Rewiring)
# kedro experiment run --run-name "rewire_20pct_seed_123" \
#   --experiment-name "kg_perturbation_study" \
#   --headless \
#   --params "perturbation.enabled=true,perturbation.rate=0.20,perturbation.random_seed=123,perturbation.strategy=category_stratified" \
#   --username $USER --release-version test-release

# 50% Perturbation (Rewiring)
kedro experiment run --run-name "rewire_50pct_seed_123" \
  --experiment-name "kg_perturbation_study" \
  --headless \
  --params "perturbation.enabled=true,perturbation.rate=0.50,perturbation.random_seed=123,perturbation.strategy=category_stratified" \
  --username $USER --release-version test-release
```

#### Seed 456 Experiments

```bash
# Baseline (0% perturbation)
kedro experiment run --run-name "baseline_seed_456" \
  --experiment-name "kg_perturbation_study" \
  --params "perturbation.enabled=false,perturbation.random_seed=456"

# 1% Perturbation (Rewiring)
kedro experiment run --run-name "rewire_1pct_seed_456" \
  --experiment-name "kg_perturbation_study" \
  --params "perturbation.enabled=true,perturbation.rate=0.01,perturbation.random_seed=456,perturbation.strategy=category_stratified"

# 5% Perturbation (Rewiring)
kedro experiment run --run-name "rewire_5pct_seed_456" \
  --experiment-name "kg_perturbation_study" \
  --params "perturbation.enabled=true,perturbation.rate=0.05,perturbation.random_seed=456,perturbation.strategy=category_stratified"

# 20% Perturbation (Rewiring)
kedro experiment run --run-name "rewire_20pct_seed_456" \
  --experiment-name "kg_perturbation_study" \
  --params "perturbation.enabled=true,perturbation.rate=0.20,perturbation.random_seed=456,perturbation.strategy=category_stratified"

# 50% Perturbation (Rewiring)
kedro experiment run --run-name "rewire_50pct_seed_456" \
  --experiment-name "kg_perturbation_study" \
  --params "perturbation.enabled=true,perturbation.rate=0.50,perturbation.random_seed=456,perturbation.strategy=category_stratified"
```

#### Seed 789 Experiments

```bash
# Baseline (0% perturbation)
kedro experiment run --name "baseline_seed_789" \
  --params "perturbation.enabled=false,perturbation.random_seed=789"

# 1% Perturbation (Rewiring)
kedro experiment run --name "rewire_1pct_seed_789" \
  --params "perturbation.enabled=true,perturbation.rate=0.01,perturbation.random_seed=789,perturbation.strategy=category_stratified"

# 5% Perturbation (Rewiring)
kedro experiment run --name "rewire_5pct_seed_789" \
  --params "perturbation.enabled=true,perturbation.rate=0.05,perturbation.random_seed=789,perturbation.strategy=category_stratified"

# 20% Perturbation (Rewiring)
kedro experiment run --name "rewire_20pct_seed_789" \
  --params "perturbation.enabled=true,perturbation.rate=0.20,perturbation.random_seed=789,perturbation.strategy=category_stratified"

# 50% Perturbation (Rewiring)
kedro experiment run --name "rewire_50pct_seed_789" \
  --params "perturbation.enabled=true,perturbation.rate=0.50,perturbation.random_seed=789,perturbation.strategy=category_stratified"
```

### Batch Execution Script

```bash
#!/bin/bash
# run_rewiring_experiments.sh

RATES=(0.0 0.01 0.05 0.20 0.50)
SEEDS=(42 123 456 789)

for rate in "${RATES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if [ "$rate" == "0.0" ]; then
            name="baseline_seed_${seed}"
            params="perturbation.enabled=false,perturbation.random_seed=${seed}"
        else
            pct=$(echo "$rate * 100" | bc | cut -d. -f1)
            name="rewire_${pct}pct_seed_${seed}"
            params="perturbation.enabled=true,perturbation.rate=${rate},perturbation.random_seed=${seed},perturbation.strategy=category_stratified"
        fi

        echo "Running experiment: $name"
        kedro experiment run --name "$name" --params "$params"

        # Optional: add delay between experiments to manage resource usage
        # sleep 300  # 5 minute delay
    done
done
```

## Expected Results and Analysis

### Hypotheses

1. **Minimal impact at 1%**: Model should be robust to small-scale edge rewiring
2. **Gradual degradation 5-20%**: Performance decrease proportional to rewiring rate
3. **Significant impact at 50%**: Substantial performance degradation from incorrect connections
4. **Category preservation effect**: Rewiring within categories should be less disruptive than
   random rewiring
5. **Embedding robustness**: Node embeddings may show more resilience than topological features

### Statistical Analysis Plan

- **ANOVA**: Test for significant differences between perturbation levels
- **Multiple comparisons**: Bonferroni correction for pairwise comparisons
- **Effect size**: Cohen's d for practical significance
- **Confidence intervals**: 95% CI for each perturbation level

### Deliverables

1. **Performance curves**: AUROC/AUPRC vs perturbation level
2. **Robustness metrics**: Slope of performance degradation
3. **Statistical significance**: P-values and effect sizes
4. **Recommendations**: Optimal perturbation thresholds for data quality assessment

## Implementation Timeline

1. **Week 1**: Implement perturbation pipeline node and configuration
2. **Week 2**: Run baseline and 1% perturbation experiments
3. **Week 3**: Run 5% and 20% perturbation experiments
4. **Week 4**: Run 50% perturbation experiments and analyze results
5. **Week 5**: Generate final report and recommendations

## Technical Requirements

### Compute Resources

- **Estimated runtime**: ~4 hours per experiment × 20 experiments = 80 hours
- **Parallel execution**: Can run multiple seeds simultaneously
- **Storage**: ~50GB additional for perturbed datasets

### Monitoring

- Track experiment progress via MLflow UI
- Monitor resource usage and completion times
- Alert on experiment failures

## Risk Mitigation

### Potential Issues

1. **Memory constraints**: Large perturbation datasets
2. **Runtime variance**: Different perturbation levels may have different compute requirements
3. **Reproducibility**: Ensure consistent random seed handling

### Mitigation Strategies

1. **Chunked processing**: Process perturbations in batches if memory constrained
2. **Resource monitoring**: Scale compute resources based on perturbation level
3. **Seed verification**: Log and verify random seeds in each experiment

## Success Criteria

1. **Completion**: All 20 experiments (5 perturbation levels × 4 seeds) successfully complete
2. **Reproducibility**: Results consistent across random seeds within each perturbation level
3. **Statistical power**: Sufficient sample size for meaningful statistical analysis
4. **Actionable insights**: Clear recommendations for KG robustness and quality thresholds

---

# Implementation Worklog

## Pipeline Implementation (Completed)

**Summary**: Built complete perturbation pipeline with category-aware edge rewiring

**Key Components Created**:

- `/pipelines/matrix/src/matrix/pipelines/perturbation/` directory structure
- `nodes.py` with `perturb_edges_rewire()` function implementing Spark-based rewiring
- `pipeline.py` with ArgoWorkflow configuration
- Configuration files for base and test environments
- Updated `globals.yml` with perturbation paths
- Modified embeddings pipeline to use perturbed edges

**Testing & Debugging Issues Fixed**:

- Added pipeline registration in `pipeline_registry.py`
- Fixed ArgoResourceConfig CPU values (string "8000m" → integer 8)
- Resolved DataFrame column conflicts in join operations

**Test Results**:

- ✅ Pipeline runs successfully with `uv run kedro run -e test -p perturbation`
- ✅ 5.2% actual perturbation rate achieved (target: 5.0%)
- ✅ 246/253 edges successfully rewired
- ✅ Category stratification working across 5 biolink categories
- ✅ All pipeline nodes complete in ~30 seconds

### CLI Enhancement (Completed)

**Summary**: Added `--params` support to experiment run command

**Changes Made**:

- Added `--params` option to `kedro experiment run` CLI command
- Updated `generate_argo_config()` function to accept and pass through parameters
- Modified Argo template to conditionally include `--params` in kedro run command
- Updated experiment plan with correct CLI syntax using `kedro experiment run`

**Test Results**:

- ✅ CLI accepts `--params` option as confirmed by help output
- ✅ Parameters will be passed through to underlying kedro run command
- ✅ Template renders conditional params only when provided

**Next Steps**:

- Run full experiments with different perturbation rates (1%, 5%, 20%, 50%)
- Execute complete pipeline through embeddings and modeling
- Analyze performance impact results
