-- Aggregate edges by upstream data source (KG) and primary knowledge source
-- This powers the source filter visualization for key nodes

WITH RECURSIVE
key_node_ids AS (
  SELECT id FROM UNNEST(SPLIT('${key_disease_ids}', ',')) AS id
  UNION ALL
  SELECT id FROM UNNEST(SPLIT('${key_drug_ids}', ',')) AS id
),

descendants AS (
  -- Start with the key node itself
  SELECT
    key_node_ids.id as key_node_id,
    key_node_ids.id as descendant_id,
    0 as depth
  FROM key_node_ids

  UNION ALL

  -- Recursively find all descendants via subclass_of edges
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

-- Get all edges involving key node descendants with node information
outgoing_edges AS (
  SELECT
    descendants.key_node_id,
    edges.upstream_data_source.list[SAFE_OFFSET(0)].element as upstream_data_source,
    pks.element as primary_knowledge_source,
    edges.subject,
    edges.predicate,
    edges.object,
    object_nodes.all_categories as other_all_categories,
    object_nodes.category as other_category
  FROM descendants
  JOIN `${project_id}.release_${bq_release_version}.edges_unified` edges
    ON edges.subject = descendants.descendant_id
  CROSS JOIN UNNEST(edges.primary_knowledge_sources.list) AS pks
  JOIN `${project_id}.release_${bq_release_version}.nodes_unified` object_nodes
    ON edges.object = object_nodes.id
),

incoming_edges AS (
  SELECT
    descendants.key_node_id,
    edges.upstream_data_source.list[SAFE_OFFSET(0)].element as upstream_data_source,
    pks.element as primary_knowledge_source,
    edges.subject,
    edges.predicate,
    edges.object,
    subject_nodes.all_categories as other_all_categories,
    subject_nodes.category as other_category
  FROM descendants
  JOIN `${project_id}.release_${bq_release_version}.edges_unified` edges
    ON edges.object = descendants.descendant_id
  CROSS JOIN UNNEST(edges.primary_knowledge_sources.list) AS pks
  JOIN `${project_id}.release_${bq_release_version}.nodes_unified` subject_nodes
    ON edges.subject = subject_nodes.id
),

all_edges AS (
  SELECT * FROM outgoing_edges
  UNION ALL
  SELECT * FROM incoming_edges
),

-- Apply same category mapping as key_nodes_category_edges.sql
key_node_edges AS (
  SELECT
    key_node_id,
    upstream_data_source,
    primary_knowledge_source,
    subject,
    predicate,
    object,
    CASE
      -- ChemicalEntity grouping
      WHEN EXISTS(
        SELECT 1 FROM UNNEST(other_all_categories.list) AS cat
        WHERE cat.element = 'biolink:ChemicalEntity'
      ) THEN 'ChemicalEntity'

      -- Gene (keep separate)
      WHEN other_category = 'biolink:Gene' THEN 'Gene'

      -- Protein (keep separate)
      WHEN EXISTS(
        SELECT 1 FROM UNNEST(other_all_categories.list) AS cat
        WHERE cat.element = 'biolink:Protein'
      ) THEN 'Protein'

      -- PhenotypicFeature (keep separate from Disease)
      -- IMPORTANT: Check primary category field BEFORE checking all_categories hierarchy
      -- PhenotypicFeature nodes have category='biolink:PhenotypicFeature' AND also have
      -- 'biolink:DiseaseOrPhenotypicFeature' in their all_categories hierarchy.
      -- By checking the specific category first, we correctly categorize them as PhenotypicFeature.
      -- The Disease check below will then catch actual Disease nodes without miscategorizing PhenotypicFeatures.
      -- NOTE: We use other_category to categorize edges by the connected node, not the key node itself.
      WHEN other_category = 'biolink:PhenotypicFeature' THEN 'PhenotypicFeature'

      -- Disease grouping
      -- Includes nodes with DiseaseOrPhenotypicFeature in their hierarchy
      -- This catches Disease nodes but NOT PhenotypicFeature nodes (which were checked above)
      WHEN EXISTS(
        SELECT 1 FROM UNNEST(other_all_categories.list) AS cat
        WHERE cat.element IN ('biolink:Disease', 'biolink:DiseaseOrPhenotypicFeature')
      ) THEN 'Disease'

      -- AnatomicalEntity grouping
      WHEN EXISTS(
        SELECT 1 FROM UNNEST(other_all_categories.list) AS cat
        WHERE cat.element = 'biolink:AnatomicalEntity'
      ) THEN 'AnatomicalEntity'

      -- BiologicalProcess grouping
      WHEN EXISTS(
        SELECT 1 FROM UNNEST(other_all_categories.list) AS cat
        WHERE cat.element IN ('biolink:BiologicalProcess', 'biolink:Pathway')
      ) THEN 'BiologicalProcess'

      -- MolecularActivity (keep separate)
      WHEN EXISTS(
        SELECT 1 FROM UNNEST(other_all_categories.list) AS cat
        WHERE cat.element = 'biolink:MolecularActivity'
      ) THEN 'MolecularActivity'

      -- GenomicEntity grouping
      WHEN EXISTS(
        SELECT 1 FROM UNNEST(other_all_categories.list) AS cat
        WHERE cat.element = 'biolink:GenomicEntity'
      ) THEN 'GenomicEntity'

      -- OrganismTaxon (keep separate)
      WHEN other_category = 'biolink:OrganismTaxon' THEN 'OrganismTaxon'

      -- Procedure (keep separate)
      WHEN other_category = 'biolink:Procedure' THEN 'Procedure'

      -- Activity/Behavior grouping
      WHEN EXISTS(
        SELECT 1 FROM UNNEST(other_all_categories.list) AS cat
        WHERE cat.element IN ('biolink:Activity', 'biolink:Behavior')
      ) THEN 'Activity'

      -- Cohort/Population grouping
      WHEN other_category IN ('biolink:Cohort', 'biolink:PopulationOfIndividualOrganisms') THEN 'Population'

      -- Everything else
      ELSE 'Other'
    END as parent_category
  FROM all_edges
)

-- Aggregate by key node, category, KG source, and primary knowledge source
SELECT
  key_node_id,
  parent_category,
  upstream_data_source,
  REPLACE(primary_knowledge_source, 'infores:', '') as primary_knowledge_source,
  COUNT(*) as edge_count,
  COUNT(DISTINCT subject) as unique_subjects,
  COUNT(DISTINCT object) as unique_objects
FROM key_node_edges
WHERE upstream_data_source IS NOT NULL
  AND primary_knowledge_source IS NOT NULL
GROUP BY key_node_id, parent_category, upstream_data_source, primary_knowledge_source
ORDER BY key_node_id, parent_category, edge_count DESC
