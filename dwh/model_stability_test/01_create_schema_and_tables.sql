-- # NOTE: This file was partially generated using AI assistance.

/*
Step 1: Drop and recreate schema and tables for model stability testing in BigQuery.
This script sets up the schema and the tables for drugs, diseases, and scores.
*/

DROP SCHEMA IF EXISTS model_stability_test CASCADE;

-- Create schema for test data
CREATE SCHEMA IF NOT EXISTS model_stability_test
  OPTIONS(
    description = 'performance testing data',
    location = 'US' -- Optional: Specify your preferred location
  );

-- Create drugs table
CREATE OR REPLACE TABLE model_stability_test.drugs AS
SELECT
  GENERATE_UUID() AS id,
  CONCAT('drug-', SUBSTR(TO_HEX(SHA256(GENERATE_UUID())), 1, 16)) AS name
FROM
  UNNEST(GENERATE_ARRAY(1, 30));

-- Create diseases table
CREATE OR REPLACE TABLE model_stability_test.diseases AS
SELECT
  GENERATE_UUID() AS id,
  CONCAT('disease-', SUBSTR(TO_HEX(SHA256(GENERATE_UUID())), 1, 16)) AS name
FROM
  UNNEST(GENERATE_ARRAY(1, 90));

-- Create scores table with model_id and model_version
CREATE OR REPLACE TABLE model_stability_test.scores (
  source STRING OPTIONS(description="Drug ID (maps to model_stability_test.drugs.id)"),
  target STRING OPTIONS(description="Disease ID (maps to model_stability_test.diseases.id)"),
  score FLOAT64 OPTIONS(description="Random score between 0 and 1"),
  run_id STRING OPTIONS(description="Identifier for the data generation run (e.g., 'run_01')"),
  model_id STRING OPTIONS(description="Model name, e.g., 'matrix'"),
  model_version STRING OPTIONS(description="Model version in semver, e.g., '0.1.3'"),
  rank INT64 OPTIONS(description="Rank of the score across all runs (1 = highest score)"),
  run_number INT64 OPTIONS(description="Run number (1-10)")
)
PARTITION BY RANGE_BUCKET(rank, GENERATE_ARRAY(1, 250000001, 50000)) -- Partition by rank in buckets of 10000
CLUSTER BY run_id, target -- Cluster by the disease ID (target)
OPTIONS(
  description="Scores table joining drugs and diseases, partitioned by rank and clustered by target (disease_id)"
); 