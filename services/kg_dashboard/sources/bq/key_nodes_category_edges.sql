-- Get example edges for a specific category connection to a key node
-- This query returns individual edge examples with subject, predicate, object
-- for drill-down when a user clicks on a category in the chord diagram

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

-- Get all edges where key node descendant is the subject (outgoing)
outgoing_edges AS (
  SELECT
    descendants.key_node_id,
    edges.subject,
    subject_nodes.name as subject_name,
    subject_nodes.category as subject_category,
    edges.predicate,
    edges.object,
    object_nodes.name as object_name,
    object_nodes.category as object_category,
    object_nodes.category as other_category,
    object_nodes.all_categories as object_all_categories,
    edges.primary_knowledge_sources as primary_knowledge_sources,
    edges.upstream_data_source.list[SAFE_OFFSET(0)].element as upstream_data_source
  FROM descendants
  JOIN `${project_id}.release_${bq_release_version}.edges_unified` edges
    ON edges.subject = descendants.descendant_id
  JOIN `${project_id}.release_${bq_release_version}.nodes_unified` subject_nodes
    ON edges.subject = subject_nodes.id
  JOIN `${project_id}.release_${bq_release_version}.nodes_unified` object_nodes
    ON edges.object = object_nodes.id
),

-- Get all edges where key node descendant is the object (incoming)
incoming_edges AS (
  SELECT
    descendants.key_node_id,
    edges.subject,
    subject_nodes.name as subject_name,
    subject_nodes.category as subject_category,
    edges.predicate,
    edges.object,
    object_nodes.name as object_name,
    object_nodes.category as object_category,
    subject_nodes.category as other_category,
    subject_nodes.all_categories as other_all_categories,
    edges.primary_knowledge_sources as primary_knowledge_sources,
    edges.upstream_data_source.list[SAFE_OFFSET(0)].element as upstream_data_source
  FROM descendants
  JOIN `${project_id}.release_${bq_release_version}.edges_unified` edges
    ON edges.object = descendants.descendant_id
  JOIN `${project_id}.release_${bq_release_version}.nodes_unified` subject_nodes
    ON edges.subject = subject_nodes.id
  JOIN `${project_id}.release_${bq_release_version}.nodes_unified` object_nodes
    ON edges.object = object_nodes.id
),

-- Combine and categorize all edges
all_edges AS (
  SELECT
    key_node_id,
    subject,
    subject_name,
    subject_category,
    predicate,
    object,
    object_name,
    object_category,
    other_category,
    object_all_categories as other_all_categories,
    primary_knowledge_sources,
    upstream_data_source
  FROM outgoing_edges

  UNION ALL

  SELECT
    key_node_id,
    subject,
    subject_name,
    subject_category,
    predicate,
    object,
    object_name,
    object_category,
    other_category,
    other_all_categories,
    primary_knowledge_sources,
    upstream_data_source
  FROM incoming_edges
),

-- Map to parent categories (same logic as key_nodes_category_summary.sql)
-- This reduces 44+ specific biolink categories to ~14 parent groups for cleaner visualization
-- We use the all_categories field to check the full category hierarchy, not just the primary category
-- This ensures we group based on the biolink inheritance tree (e.g., Drug is a ChemicalEntity)
categorized_edges AS (
  SELECT
    key_node_id,
    subject,
    subject_name,
    subject_category,
    predicate,
    object,
    object_name,
    object_category,
    primary_knowledge_sources,
    upstream_data_source,
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
),

-- Extract a single primary source for sampling purposes
edges_with_primary_source AS (
  SELECT
    *,
    -- Extract first primary knowledge source for sampling
    (SELECT REPLACE(pks.element, 'infores:', '') FROM UNNEST(primary_knowledge_sources.list) AS pks LIMIT 1) as primary_source
  FROM categorized_edges
)

-- Return example edges for ALL categories (we'll filter in JavaScript)
-- Limited to 10 edges per primary_knowledge_source per category per key node
-- This ensures diverse representation across all knowledge sources
SELECT
  key_node_id,
  parent_category,
  subject,
  subject_name,
  REPLACE(predicate, 'biolink:', '') as predicate,
  object,
  object_name,
  ARRAY_TO_STRING(
    ARRAY(
      SELECT REPLACE(pks.element, 'infores:', '')
      FROM UNNEST(primary_knowledge_sources.list) AS pks
    ),
    ', '
  ) as primary_knowledge_sources,
  upstream_data_source,
  ROW_NUMBER() OVER (
    PARTITION BY key_node_id, parent_category, primary_source
    ORDER BY subject, predicate, object
  ) as row_num
FROM edges_with_primary_source
QUALIFY row_num <= 10
