---
title: FAQ
---

## How do I get access to GCP?

## Where do we store shared datasets?

## I get ```ERROR Neo4jDataWriter: Cannot commit the transaction because: Node(479) already exists with label `Entity` and property `id` = 'node_1' ```error
This is because you didn't clean your database state; run `Make wipe_neo` and try again.

## I get ```xgboost.core.XGBoostError: XGBoost Library (libxgboost.dylib) could not be loaded.``` error when running ```kedro run -p test```
Likely due to OpenMP runtime not being installed. For Mac OSX users: Run `brew install libomp` to install OpenMP runtime.
...
