-- # NOTE: This file was partially generated using AI assistance.

/*
Step 4: Run the stability metric for each consecutive run pair (01/02, 02/03, ..., 09/10).
This script outputs the stability metric for each pair.
*/

WITH run_pairs AS (
  SELECT 'run_01' AS run_id_1, 'run_02' AS run_id_2 UNION ALL
  SELECT 'run_02', 'run_03' UNION ALL
  SELECT 'run_03', 'run_04' UNION ALL
  SELECT 'run_04', 'run_05' UNION ALL
  SELECT 'run_05', 'run_06' UNION ALL
  SELECT 'run_06', 'run_07' UNION ALL
  SELECT 'run_07', 'run_08' UNION ALL
  SELECT 'run_08', 'run_09' UNION ALL
  SELECT 'run_09', 'run_10'
)
SELECT
  rp.run_id_1,
  rp.run_id_2,
  CORR(s1.score, s2.score) AS stability
FROM run_pairs rp
JOIN model_stability_test.scores s1 ON s1.run_id = rp.run_id_1
JOIN model_stability_test.scores s2 ON s2.run_id = rp.run_id_2
  AND s1.source = s2.source AND s1.target = s2.target
GROUP BY rp.run_id_1, rp.run_id_2
ORDER BY rp.run_id_1, rp.run_id_2; 