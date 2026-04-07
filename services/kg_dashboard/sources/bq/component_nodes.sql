-- Nodes for the 50 largest minor components plus any with core entities
-- Used by the component detail parameterized page
WITH top_by_size AS (
  SELECT component_id
  FROM `${project_id}.release_${bq_release_version}.connected_components`
  WHERE component_id > 0
  ORDER BY component_size DESC
  LIMIT 50
),
with_core AS (
  SELECT component_id
  FROM `${project_id}.release_${bq_release_version}.connected_components`
  WHERE component_id > 0
    AND (num_drugs > 0 OR num_diseases > 0)
),
target_components AS (
  SELECT component_id FROM top_by_size
  UNION DISTINCT
  SELECT component_id FROM with_core
)
SELECT
  nm.component_id,
  nm.id,
  n.name,
  REPLACE(n.category, 'biolink:', '') as category,
  nm.ec_core_category,
  CASE
    WHEN nm.ec_core_category = 'drug' THEN 'Core Drugs'
    WHEN nm.ec_core_category = 'disease' THEN 'Core Diseases'

    WHEN EXISTS(
      SELECT 1 FROM UNNEST(n.all_categories.list) AS cat
      WHERE cat.element = 'biolink:ChemicalEntity'
    ) THEN 'ChemicalEntity'

    WHEN n.category = 'biolink:Gene' THEN 'Gene'

    WHEN n.category = 'biolink:Protein' OR EXISTS(
      SELECT 1 FROM UNNEST(n.all_categories.list) AS cat
      WHERE cat.element = 'biolink:Protein'
    ) THEN 'Protein'

    WHEN EXISTS(
      SELECT 1 FROM UNNEST(n.all_categories.list) AS cat
      WHERE cat.element IN ('biolink:Disease', 'biolink:DiseaseOrPhenotypicFeature')
    ) AND n.category = 'biolink:Disease' THEN 'Disease'

    WHEN n.category = 'biolink:PhenotypicFeature' THEN 'PhenotypicFeature'

    WHEN EXISTS(
      SELECT 1 FROM UNNEST(n.all_categories.list) AS cat
      WHERE cat.element = 'biolink:AnatomicalEntity'
    ) THEN 'AnatomicalEntity'

    WHEN EXISTS(
      SELECT 1 FROM UNNEST(n.all_categories.list) AS cat
      WHERE cat.element IN ('biolink:BiologicalProcess', 'biolink:Pathway')
    ) THEN 'BiologicalProcess'

    WHEN n.category = 'biolink:MolecularActivity' OR EXISTS(
      SELECT 1 FROM UNNEST(n.all_categories.list) AS cat
      WHERE cat.element = 'biolink:MolecularActivity'
    ) THEN 'MolecularActivity'

    WHEN EXISTS(
      SELECT 1 FROM UNNEST(n.all_categories.list) AS cat
      WHERE cat.element = 'biolink:GenomicEntity'
    ) THEN 'GenomicEntity'

    WHEN n.category = 'biolink:OrganismTaxon' THEN 'OrganismTaxon'
    WHEN n.category = 'biolink:Procedure' THEN 'Procedure'

    WHEN EXISTS(
      SELECT 1 FROM UNNEST(n.all_categories.list) AS cat
      WHERE cat.element IN ('biolink:Activity', 'biolink:Behavior')
    ) THEN 'Activity'

    WHEN n.category IN ('biolink:Cohort', 'biolink:PopulationOfIndividualOrganisms') THEN 'Population'

    ELSE 'Other'
  END as parent_category
FROM `${project_id}.release_${bq_release_version}.node_metrics` nm
JOIN `${project_id}.release_${bq_release_version}.nodes_unified` n ON nm.id = n.id
WHERE nm.component_id IN (SELECT component_id FROM target_components)
ORDER BY nm.component_id, n.category, n.name
