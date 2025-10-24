-- Category connections summary for key nodes (direction-agnostic)
-- Groups all edges by parent category using biolink hierarchy from all_categories

WITH RECURSIVE
key_node_ids AS (
  SELECT id FROM UNNEST(SPLIT('${key_disease_ids}', ',')) AS id
  UNION ALL
  SELECT id FROM UNNEST(SPLIT('${key_drug_ids}', ',')) AS id
),

descendants AS (
  SELECT
    key_node_ids.id as key_node_id,
    key_node_ids.id as descendant_id,
    0 as depth
  FROM key_node_ids

  UNION ALL

  SELECT
    descendants.key_node_id,
    edges.subject as descendant_id,
    descendants.depth + 1 as depth
  FROM descendants
  JOIN `${project_id}.release_${bq_release_version}.edges_unified` edges
    ON edges.object = descendants.descendant_id
    AND edges.predicate = 'biolink:subclass_of'
  WHERE descendants.depth < 20
    AND EXISTS(
      SELECT 1 FROM UNNEST(edges.primary_knowledge_sources.list) AS pks
      WHERE pks.element IN ('infores:mondo', 'infores:chebi')
    )
),

-- Get edges where key node descendant is the subject (outgoing)
outgoing_edges AS (
  SELECT
    descendants.key_node_id,
    object_nodes.id as connected_node_id,
    object_nodes.category as connected_category,
    object_nodes.all_categories as all_categories
  FROM descendants
  JOIN `${project_id}.release_${bq_release_version}.edges_unified` edges
    ON edges.subject = descendants.descendant_id
  JOIN `${project_id}.release_${bq_release_version}.nodes_unified` object_nodes
    ON edges.object = object_nodes.id
),

-- Get edges where key node descendant is the object (incoming)
incoming_edges AS (
  SELECT
    descendants.key_node_id,
    subject_nodes.id as connected_node_id,
    subject_nodes.category as connected_category,
    subject_nodes.all_categories as all_categories
  FROM descendants
  JOIN `${project_id}.release_${bq_release_version}.edges_unified` edges
    ON edges.object = descendants.descendant_id
  JOIN `${project_id}.release_${bq_release_version}.nodes_unified` subject_nodes
    ON edges.subject = subject_nodes.id
),

-- Combine both directions
all_edges AS (
  SELECT * FROM outgoing_edges
  UNION ALL
  SELECT * FROM incoming_edges
),

-- Map to parent categories based on biolink hierarchy
categorized_edges AS (
  SELECT
    key_node_id,
    connected_node_id,
    CASE
      -- ChemicalEntity grouping
      WHEN EXISTS(
        SELECT 1 FROM UNNEST(all_categories.list) AS cat
        WHERE cat.element = 'biolink:ChemicalEntity'
      ) THEN 'ChemicalEntity'

      -- Gene (keep separate, important for drug repurposing)
      WHEN connected_category = 'biolink:Gene' THEN 'Gene'

      -- Protein (keep separate, important for drug repurposing)
      WHEN connected_category = 'biolink:Protein' OR EXISTS(
        SELECT 1 FROM UNNEST(all_categories.list) AS cat
        WHERE cat.element = 'biolink:Protein'
      ) THEN 'Protein'

      -- Disease grouping
      WHEN EXISTS(
        SELECT 1 FROM UNNEST(all_categories.list) AS cat
        WHERE cat.element IN ('biolink:Disease', 'biolink:DiseaseOrPhenotypicFeature')
      ) AND connected_category = 'biolink:Disease' THEN 'Disease'

      -- PhenotypicFeature (keep separate from Disease)
      WHEN connected_category = 'biolink:PhenotypicFeature' THEN 'PhenotypicFeature'

      -- AnatomicalEntity grouping
      WHEN EXISTS(
        SELECT 1 FROM UNNEST(all_categories.list) AS cat
        WHERE cat.element = 'biolink:AnatomicalEntity'
      ) THEN 'AnatomicalEntity'

      -- BiologicalProcess grouping
      WHEN EXISTS(
        SELECT 1 FROM UNNEST(all_categories.list) AS cat
        WHERE cat.element IN ('biolink:BiologicalProcess', 'biolink:Pathway')
      ) THEN 'BiologicalProcess'

      -- MolecularActivity (keep separate)
      WHEN connected_category = 'biolink:MolecularActivity' OR EXISTS(
        SELECT 1 FROM UNNEST(all_categories.list) AS cat
        WHERE cat.element = 'biolink:MolecularActivity'
      ) THEN 'MolecularActivity'

      -- GenomicEntity grouping
      WHEN EXISTS(
        SELECT 1 FROM UNNEST(all_categories.list) AS cat
        WHERE cat.element = 'biolink:GenomicEntity'
      ) THEN 'GenomicEntity'

      -- OrganismTaxon (keep separate)
      WHEN connected_category = 'biolink:OrganismTaxon' THEN 'OrganismTaxon'

      -- Procedure (keep separate for clinical context)
      WHEN connected_category = 'biolink:Procedure' THEN 'Procedure'

      -- Activity/Behavior grouping
      WHEN EXISTS(
        SELECT 1 FROM UNNEST(all_categories.list) AS cat
        WHERE cat.element IN ('biolink:Activity', 'biolink:Behavior')
      ) THEN 'Activity'

      -- Cohort/Population grouping
      WHEN connected_category IN ('biolink:Cohort', 'biolink:PopulationOfIndividualOrganisms') THEN 'Population'

      -- Everything else
      ELSE 'Other'
    END as parent_category
  FROM all_edges
)

-- Count edges and distinct nodes by parent category
SELECT
  key_node_id,
  parent_category as connected_category,
  COUNT(DISTINCT connected_node_id) as distinct_nodes,
  COUNT(*) as total_edges
FROM categorized_edges
GROUP BY key_node_id, parent_category
ORDER BY key_node_id, distinct_nodes DESC
