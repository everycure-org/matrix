-- # NOTE: This file was partially generated using AI assistance.

/*
Step 2: Insert synthetic scores for all runs into the model_stability_test.scores table.
- Runs 1-3: Completely random scores (RAND())
- Runs 4-6: Deterministic, identical scores for each drug-disease pair (hash-based)
- Runs 7-10: Highly correlated scores, seeded from the drug-disease pair and slightly perturbed by run number
*/

TRUNCATE TABLE model_stability_test.scores;

INSERT INTO model_stability_test.scores (source, target, score, run_id, model_id, model_version, rank, run_number)
SELECT
  source,
  target,
  score,
  run_id,
  model_id,
  model_version,
  RANK() OVER (ORDER BY score DESC) AS rank,
  run_number
FROM (
  SELECT
    source,
    target,
    -- Calculate the score in this subquery
    CASE
      WHEN run_number BETWEEN 1 AND 3 THEN RAND()
      WHEN run_number BETWEEN 4 AND 6 THEN
        ABS(FARM_FINGERPRINT(CONCAT(source, target))) / 9223372036854775807
      WHEN run_number = 7 THEN
        0.6 * (ABS(FARM_FINGERPRINT(CONCAT(source, target))) / 9223372036854775807) + 0.4 * RAND()
      WHEN run_number = 8 THEN
        0.7 * (ABS(FARM_FINGERPRINT(CONCAT(source, target))) / 9223372036854775807) + 0.3 * RAND()
      WHEN run_number = 9 THEN
        0.8 * (ABS(FARM_FINGERPRINT(CONCAT(source, target))) / 9223372036854775807) + 0.2 * RAND()
      WHEN run_number = 10 THEN
        0.9 * (ABS(FARM_FINGERPRINT(CONCAT(source, target))) / 9223372036854775807) + 0.1 * RAND()
    END AS score,
    FORMAT('run_%02d', run_number) AS run_id,
    'matrix' AS model_id,
    CASE MOD(run_number, 4)
      WHEN 0 THEN '0.1.3'
      WHEN 1 THEN '0.2.0'
      WHEN 2 THEN '0.3.5'
      ELSE '0.4.1'
    END AS model_version,
    run_number
  FROM (
    SELECT
      d.id AS source,
      dis.id AS target,
      run_number
    FROM
      model_stability_test.drugs AS d
    CROSS JOIN
      model_stability_test.diseases AS dis
    CROSS JOIN
      UNNEST(GENERATE_ARRAY(1, 10)) AS run_number
  )
); 