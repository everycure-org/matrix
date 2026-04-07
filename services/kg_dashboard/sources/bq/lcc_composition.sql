-- LCC node composition: core entities (drug/disease) + non-core grouped by parent category
-- Pre-aggregated to ~15 rows for dashboard display

WITH lcc_nodes AS (
  SELECT
    nm.id,
    nm.ec_core_category,
    n.category,
    n.all_categories
  FROM `${project_id}.release_${bq_release_version}.node_metrics` nm
  JOIN `${project_id}.release_${bq_release_version}.nodes_unified` n ON nm.id = n.id
  WHERE nm.component_id = 0
),

categorized AS (
  SELECT
    CASE
      WHEN ec_core_category = 'drug' THEN 'Core Drugs'
      WHEN ec_core_category = 'disease' THEN 'Core Diseases'

      WHEN EXISTS(
        SELECT 1 FROM UNNEST(all_categories.list) AS cat
        WHERE cat.element = 'biolink:ChemicalEntity'
      ) THEN 'ChemicalEntity'

      WHEN category = 'biolink:Gene' THEN 'Gene'

      WHEN category = 'biolink:Protein' OR EXISTS(
        SELECT 1 FROM UNNEST(all_categories.list) AS cat
        WHERE cat.element = 'biolink:Protein'
      ) THEN 'Protein'

      WHEN EXISTS(
        SELECT 1 FROM UNNEST(all_categories.list) AS cat
        WHERE cat.element IN ('biolink:Disease', 'biolink:DiseaseOrPhenotypicFeature')
      ) AND category = 'biolink:Disease' THEN 'Disease'

      WHEN category = 'biolink:PhenotypicFeature' THEN 'PhenotypicFeature'

      WHEN EXISTS(
        SELECT 1 FROM UNNEST(all_categories.list) AS cat
        WHERE cat.element = 'biolink:AnatomicalEntity'
      ) THEN 'AnatomicalEntity'

      WHEN EXISTS(
        SELECT 1 FROM UNNEST(all_categories.list) AS cat
        WHERE cat.element IN ('biolink:BiologicalProcess', 'biolink:Pathway')
      ) THEN 'BiologicalProcess'

      WHEN category = 'biolink:MolecularActivity' OR EXISTS(
        SELECT 1 FROM UNNEST(all_categories.list) AS cat
        WHERE cat.element = 'biolink:MolecularActivity'
      ) THEN 'MolecularActivity'

      WHEN EXISTS(
        SELECT 1 FROM UNNEST(all_categories.list) AS cat
        WHERE cat.element = 'biolink:GenomicEntity'
      ) THEN 'GenomicEntity'

      WHEN category = 'biolink:OrganismTaxon' THEN 'OrganismTaxon'
      WHEN category = 'biolink:Procedure' THEN 'Procedure'

      WHEN EXISTS(
        SELECT 1 FROM UNNEST(all_categories.list) AS cat
        WHERE cat.element IN ('biolink:Activity', 'biolink:Behavior')
      ) THEN 'Activity'

      WHEN category IN ('biolink:Cohort', 'biolink:PopulationOfIndividualOrganisms') THEN 'Population'

      ELSE 'Other'
    END as parent_category,
    CASE WHEN ec_core_category IS NOT NULL THEN true ELSE false END as is_core
  FROM lcc_nodes
)

SELECT
  parent_category,
  is_core,
  COUNT(*) as node_count
FROM categorized
GROUP BY parent_category, is_core
ORDER BY node_count DESC
