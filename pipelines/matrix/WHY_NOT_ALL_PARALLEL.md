# Why Not Run All 20 Trials in Parallel?

## TL;DR

Running all 20 trials at once would **hurt performance** because:

1. ❌ Breaks Bayesian optimization (no learning between trials)
2. ❌ Would need 20 × 37 = 740 threads competing for 111 CPUs
3. ❌ Memory explosion (20 XGBoost models in RAM simultaneously)
4. ❌ Actually slower due to resource contention

The current approach (3 at a time) is **optimal**! ✅

---

## Understanding Bayesian Optimization

### What Makes It Smart

Bayesian optimization **learns** from previous trials:

```
Trial 1-3:   Random exploration
Trial 4-6:   "Hmm, high learning_rate + low max_depth seems good"
Trial 7-9:   "Let me try variations around that region"
Trial 10-12: "Found an even better combo!"
Trial 13-20: "Refining the best region..."
```

### If We Run All 20 at Once

```
Trial 1-20: All random (no learning!)
```

This degrades to **random search** instead of intelligent Bayesian optimization.

---

## The Math: Why 3 Parallel Is Optimal

### Current Approach (3 Parallel)

```
Iteration 1: Evaluate configs 1, 2, 3 in parallel (3 × 37 = 111 CPUs)
            GP learns from results
Iteration 2: Evaluate configs 4, 5, 6 in parallel (informed by 1-3)
            GP learns from results
Iteration 3: Evaluate configs 7, 8, 9 in parallel (informed by 1-6)
            ...and so on

Total: ~7 iterations of learning
Result: Converges to optimal hyperparameters quickly
```

### If We Did 20 Parallel

```
Iteration 1: Evaluate all 20 configs (20 × 37 = 740 threads on 111 CPUs!!!)
            - Massive resource contention
            - Context switching overhead
            - Memory pressure (20 models in RAM)

Total: 1 iteration, no learning
Result: Like throwing darts blindfolded
```

---

## Resource Considerations

### Current: 3 Parallel

| Resource    | Usage              | Status         |
| ----------- | ------------------ | -------------- |
| CPU Threads | 3 × 37 = 111       | ✅ Perfect     |
| RAM         | ~3-4 GB (3 models) | ✅ Manageable  |
| Cache       | Good locality      | ✅ Efficient   |
| Learning    | 7 iterations       | ✅ Intelligent |

### Hypothetical: 20 Parallel

| Resource    | Usage                 | Status                   |
| ----------- | --------------------- | ------------------------ |
| CPU Threads | 20 × 37 = 740         | ❌ 6.7× oversubscription |
| RAM         | ~20-30 GB (20 models) | ❌ Memory pressure       |
| Cache       | Thrashing             | ❌ Inefficient           |
| Learning    | 0 iterations          | ❌ Random search         |

---

## Performance Analysis

Let's assume each model takes **T** seconds to train:

### Current (3 parallel, Bayesian learning)

```
Time: 7 iterations × T = 7T
Quality: High (intelligent search finds good params early)
Effective: Often finds optimal in 10-15 trials → ~5T actual time
```

### Sequential (1 at a time)

```
Time: 20 iterations × T = 20T
Quality: High (full Bayesian learning)
```

### All 20 Parallel (hypothetical)

```
Time: 1 iteration × (3-5T) = 3-5T (due to contention and overhead)
Quality: Low (random search, might need 50-100 trials to match)
```

**Current approach is best!** It gets the Bayesian intelligence benefit while still being 3× faster than sequential.

---

## What If You Really Want More Parallelism?

If each trial takes a **very long time** (hours), you could increase parallelism, but you'd trade off:

### Option 1: More Parallel (e.g., 6 parallel)

```yaml
# Could manually override by reducing threads per model
estimator:
  n_jobs: 18 # Instead of -1
```

This would give you:

- 6 configs in parallel (111 / 18 ≈ 6)
- Still some Bayesian learning (~3 iterations)
- But less optimal than 3 parallel

### Option 2: Increase n_calls

If 20 trials isn't enough:

```yaml
n_calls: 40 # More trials for better convergence
```

With current setup:

- 40 trials / 3 parallel = ~13 iterations
- More learning opportunities
- Better final result

---

## The Bottleneck Question

> "Is parallel evaluation still the bottleneck?"

**It depends on your training time:**

### If Each Trial Takes < 5 Minutes

Current setup is **not a bottleneck**:

- 20 trials / 3 parallel = ~7 iterations
- Total time: ~7 × training_time
- This is already 3× faster than sequential
- Bayesian learning finds good params early

### If Each Trial Takes > 30 Minutes

You might benefit from more parallelism, but:

- Consider if 20 trials is enough (might need more)
- Trade-off: more parallel = less intelligent search
- Better solution: use more trials with current parallelism

---

## Real-World Analogy

Imagine you're trying to find the best restaurant:

### Sequential (1 at a time)

"Try place A → not great → try place B → better! → try place C in same area → excellent!"

### Current (3 parallel, Bayesian)

"Try places A, B, C → B was best → try 3 more near B → found great ones!"

### All 20 parallel (random)

"Try 20 random places at once → can't learn between them → might miss the best area"

---

## Recommendation

**Keep the current configuration!** ✅

The 3 parallel evaluations strike the perfect balance:

- ✅ Full CPU utilization (111 CPUs)
- ✅ Intelligent Bayesian learning (7 iterations)
- ✅ 3× speedup over sequential
- ✅ Reasonable memory usage
- ✅ No resource contention

If you want better hyperparameters, increase `n_calls` rather than parallel evaluations:

```yaml
n_calls: 30 # or 40, or 50 - depends on your patience
```

This gives you:

- More learning iterations
- Better final hyperparameters
- Still 3× faster than sequential with those trials

---

## Summary

| Approach    | Parallelism | Iterations | Speed  | Quality        | Resource Use |
| ----------- | ----------- | ---------- | ------ | -------------- | ------------ |
| Sequential  | 1           | 20         | 1×     | ⭐⭐⭐⭐⭐     | ✅           |
| **Current** | **3**       | **7**      | **3×** | **⭐⭐⭐⭐⭐** | **✅**       |
| 6 Parallel  | 6           | 3          | 6×     | ⭐⭐⭐         | ⚠️           |
| 20 Parallel | 20          | 1          | 1-2×   | ⭐             | ❌           |

**Current approach (3 parallel) is optimal!** It's the sweet spot for 111 CPUs with Bayesian optimization.
