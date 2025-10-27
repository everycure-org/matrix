# ADR: Make modelling/tuning CPU‑first; remove GPU dependency in the pipeline

Status: Approved
Date: 2025‑10‑27
Deciders: AI Platform & Machine Learning Team (Team Data and Pair Predictions)

# Context

Our cloud account is hard‑capped at 16 GPUs globally. Capacity is on‑demand (not reserved), so availability is volatile; in practice we often see only 5–6 GPUs available.

We recently increased shards from 3 to 6, which (with 3‑fold) creates 18 concurrently schedulable modelling nodes. When those nodes requested GPUs, they frequently blocked on scheduling, stalling multiple pipelines behind scarce GPU capacity.

CPU capacity is abundant and inexpensive via Spot, and our XGBoost‑based modelling/tuning can exploit CPU parallelism efficiently (details below).

The associated PR demonstrates that removing GPU requirements and running the stage on high‑CPU Spot nodes improves both node and end‑to‑end pipeline times without sacrificing model quality.

# Decision

- Remove GPU requests/limits from the modelling/tuning stage of the pipeline (XGBoost ensemble and hyperparameter search). Apply CPU‑first scheduling on large CPU Spot pools. Implemented in PR #1869.

- Adopt adaptive CPU allocation: keep n_jobs=-1 for XGBoost; allow the tuner to auto‑detect CPUs, select a small number of parallel evaluations (≈2–4), and assign threads per model to avoid over‑subscription, maximizing CPU utilization and stability.

- Constrain heavy parallelism at the tuning layer (not by multiplying GPU‑bound pods). The Gaussian-process tuner uses the ask/tell pattern to evaluate a handful of trials in parallel while learning between iterations, avoiding the combinatorial thread explosion that previously caused contention.

- Keep GPU auto‑detection in Spark utilities for future workloads that may benefit from accelerators; default remains CPU‑only when GPUs are absent.

- Operational guardrails: set pipeline‑level concurrency so that Shard×Fold fan‑out does not exceed available CPU headroom on Spot pools; queuing happens inside the CPU‑rich pool rather than at the global GPU gate.

# Rationale

**Removes the bottleneck**: GPU scarcity was the systemic queueing point; CPU Spot capacity scales elastically.

**Better throughput**: With adaptive outer/inner parallelism on CPUs, we fully utilize cores with minimal contention and maintain Bayesian optimization quality.

**Cost & availability**: CPU Spot is significantly cheaper and far more available than on‑demand GPUs in our regions (and GPU Spot isn’t reliably available to us).

- ~50% faster for the Shard×Fold modelling node completion.

- ~70% faster overall pipeline runtime in representative runs, executed fully on Spot with no GPUs.

These improvements stem from

- elimination of GPU scheduling waits
- efficient CPU utilization via the tuner’s parallelization strategy.

The CPU allocation logic explicitly balances outer parallelism (number of configs evaluated at once) with inner parallelism (XGBoost threads per model) using a capped √CPU heuristic, keeping n_jobs=-1 in config while temporarily setting an effective per‑model thread count during execution.

# Implementation Notes

**Infrastructure**: PR adds/uses high‑CPU “h3‑standard‑88” node pools (regular & Spot), adjusts zone placement/taints, and enables flexible scheduling for reliability and scale.

**Modelling**: Keep n_jobs=-1 in the estimator config; the tuner will compute n_parallel_evals in [2..4] and pick threads_per_model = n_cpus / n_parallel_evals at runtime; logging confirms the chosen values during runs.

**Tuning engine**: Switch from gp_minimize to Optimizer ask/tell with controlled parallel batches; respects per‑model n_jobs and avoids degrading BO into random search.

**Spark jobs**: keep GPU auto‑config paths, but default to CPU settings when no GPUs are detected.

# Consequences

## Positive

- Removes the global GPU gate → unblocks parallel pipelines.

- Faster modelling/tuning and lower cost variability on Spot.

- Simpler capacity planning: cores are fungible across the fleet.

## Negative / Trade‑offs

- Individual XGBoost fits may be slower than on a large GPU, but end‑to‑end is faster thanks to eliminated queueing and better CPU packing.

- Spot preemption risk on CPU nodes; mitigated by retry policies and batch sizing.

- Requires maintaining the CPU allocation/tuner logic as cluster sizes evolve.

## Alternatives Considered

- Request higher GPU quotas / dedicated capacity: Long approval cycles and higher cost; still a single scarce resource.

- Keep GPUs but revert shards from 6 → 3: Reduces concurrency but limits modelling throughput and doesn’t address systemic scarcity.

- Hybrid (GPU only for specific models): Adds operational complexity now; we retain GPU detection for future targeted use.

# Metrics to Watch

- Pipeline end‑to‑end duration (p50/p90).
- Node wait time before start (should drop without GPU gating).
- CPU utilization and preemption rate on Spot pools.
- Cost per successful pipeline.
- Model quality metrics relative to prior baselines.

# Backout Plan

Re‑enable prior GPU requests/limits and reduce Shard×Fold fan‑out to 3×3 if CPU Spot becomes constrained or if regression appears.

Keep Spark GPU auto‑config available for rapid reversion to GPU‑accelerated paths.
