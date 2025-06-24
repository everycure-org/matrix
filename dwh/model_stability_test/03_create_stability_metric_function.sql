-- # NOTE: This file was partially generated using AI assistance.

/*
Step 3: Create a reusable function to calculate the stability metric (Pearson correlation) between two runs' scores.
Args:
  run_id_1: STRING, e.g., 'run_01'
  run_id_2: STRING, e.g., 'run_02'
Returns:
  Stability metric value (FLOAT64)
*/

CREATE OR REPLACE FUNCTION model_stability_test.stability_metric(run_id_1 STRING, run_id_2 STRING)
RETURNS FLOAT64 AS (
  (
    SELECT CORR(s1.score, s2.score)
    FROM model_stability_test.scores s1
    JOIN model_stability_test.scores s2
      ON s1.source = s2.source AND s1.target = s2.target
    WHERE s1.run_id = run_id_1 AND s2.run_id = run_id_2
  )
); 