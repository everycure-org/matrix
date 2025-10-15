# Parallel Hyperparameter Optimization Implementation

## Overview

The `GaussianSearch` tuner has been updated to support parallel evaluation of hyperparameter configurations using scikit-optimize's `Optimizer` class with the ask/tell pattern.

## Changes Made

### 1. Modified `tuning.py`

**Previous Implementation:**

- Used `gp_minimize` which evaluates hyperparameter configurations sequentially
- `n_jobs=-1` parameter only parallelized the acquisition function optimization (finding next point)
- 20 model fits ran one after another

**New Implementation:**

- Uses `Optimizer` with `ask`/`tell` pattern
- Enables parallel evaluation of multiple hyperparameter configurations simultaneously
- Respects the `n_jobs` parameter from the underlying estimator (e.g., XGBoost)
- When `n_jobs=-1`, multiple configurations are evaluated in parallel using `joblib.Parallel`

### 2. Key Features

- **Adaptive Parallelization**: Automatically detects `n_jobs` from the estimator
  - If `n_jobs=-1`: Evaluates multiple hyperparameter configs in parallel
  - Otherwise: Falls back to sequential evaluation
- **Bayesian Optimization**: Still uses Gaussian Process for intelligent hyperparameter search

  - `base_estimator="GP"`: Gaussian Process surrogate model
  - `acq_func="gp_hedge"`: Adaptive acquisition function selection
  - `acq_optimizer="lbfgs"`: L-BFGS for acquisition function optimization

- **Backward Compatible**: Works with existing configuration files

### 3. Configuration

No changes required to `xg_ensemble.yml`. The tuner now automatically:

1. Detects `n_jobs=-1` from the XGBoost estimator
2. Evaluates multiple hyperparameter configurations in parallel
3. Each individual XGBoost training still uses all cores (via XGBoost's `n_jobs=-1`)

### 4. Performance Impact

**Before:**

- 20 hyperparameter trials × 1 CV split = 20 sequential XGBoost fits
- Each fit uses all CPU cores for tree building

**After:**

- 20 trials evaluated in batches of N (where N = number of CPU cores)
- Each batch evaluates N configurations in parallel
- Each XGBoost fit still uses all CPU cores
- Estimated speedup: Up to N× faster (in practice 2-4× due to overhead)

## Testing

Added `test_gaussian_search_parallel()` to verify:

- Parallel evaluation works correctly
- Best parameters are found and within bounds
- Convergence plot is generated
- Compatible with sklearn's cross-validation splitters

## Example Usage

```python
from matrix.pipelines.modelling.tuning import GaussianSearch
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from skopt.space import Real, Integer

# Create estimator with n_jobs=-1 for parallel evaluation
estimator = XGBClassifier(n_jobs=-1, random_state=42)

# Define search space
dimensions = [
    Real(name='learning_rate', low=0.01, high=0.5),
    Integer(name='max_depth', low=3, high=10),
]

# Create tuner (will automatically use parallel evaluation)
tuner = GaussianSearch(
    estimator=estimator,
    dimensions=dimensions,
    scoring=lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
    splitter=StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42),
    n_calls=20
)

# Fit - will evaluate multiple configs in parallel
tuner.fit(X, y)

# Get best parameters
print(tuner.best_params_)
```

## Notes

- The implementation uses `joblib.Parallel` which respects the MATRIX project's existing parallelization patterns
- Gaussian Process surrogate model training is still sequential (required for Bayesian optimization)
- This change primarily speeds up the expensive model evaluation step
- For datasets with long training times, the speedup will be more noticeable
