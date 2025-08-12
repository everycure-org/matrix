#!/bin/bash
RELEASE_VERSION="v0.9.2"
EXPERIMENT_NAME="kg_perturbation_study"
DATETIME=$(date +%Y%m%d-%H%M%S)
# gen random suffix every time

# # Baseline (0% perturbation)
# .venv/bin/kedro experiment run --run-name "baseline_${DATETIME}" \
#   --experiment-name $EXPERIMENT_NAME \
#   --headless -q \
#   --pipeline feature_and_modelling_run \
#   --params "perturbation.enabled=false,perturbation.rate=0.0" \
#   --username $USER --release-version $RELEASE_VERSION
# 
# # # 1% Perturbation (Rewiring)
# .venv/bin/kedro experiment run --run-name "rewire_1pct_${DATETIME}" \
#   --experiment-name $EXPERIMENT_NAME \
#   --headless -q \
#   --pipeline feature_and_modelling_run \
#   --params "perturbation.enabled=true,perturbation.rate=0.01" \
#   --username $USER --release-version $RELEASE_VERSION
# 
# # 20% Perturbation (Rewiring)
# .venv/bin/kedro experiment run --run-name "rewire_20pct_${DATETIME}" \
#   --experiment-name $EXPERIMENT_NAME \
#   --headless -q \
#   --pipeline feature_and_modelling_run \
#   --params "perturbation.enabled=true,perturbation.rate=0.20" \
#   --username $USER --release-version $RELEASE_VERSION
# 
# # 50% Perturbation (Rewiring)
# .venv/bin/kedro experiment run --run-name "rewire_50pct_${DATETIME}" \
#   --experiment-name $EXPERIMENT_NAME \
#   --headless -q \
#   --pipeline feature_and_modelling_run \
#   --params "perturbation.enabled=true,perturbation.rate=0.50" \
#   --username $USER --release-version $RELEASE_VERSION

# 99% Perturbation (Rewiring)
.venv/bin/kedro experiment run --run-name "rewire_99pct_${DATETIME}" \
  --experiment-name $EXPERIMENT_NAME \
  --headless -q \
  --pipeline feature_and_modelling_run \
  --params "perturbation.enabled=true,perturbation.rate=0.99" \
  --username $USER --release-version $RELEASE_VERSION