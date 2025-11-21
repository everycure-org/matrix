# Disease List Pipeline

## Overview

The Disease List Pipeline creates the MATRIX disease list from the MONDO disease ontology. This curated list determines which diseases are included in the MATRIX drug repurposing platform. The pipeline processes raw ontology data, enriches it with external mappings, applies sophisticated filters, and generates the final validated disease list.

## How to Run

```bash
# Run the full pipeline
uv run kedro run -p disease_list
```

## Pipeline Architecture

The pipeline consists of 4 main nodes that process data sequentially:

```
Raw Data Sources → Data Preparation → MONDO Extraction → Disease List Creation
```

## Input Data Sources

- **MONDO Ontology** (`disease_list.raw.mondo_graph`): The primary disease ontology source (PyOxigraph RDF graph)
- **MONDO SSSOM Mappings** (`disease_list.raw.mondo_sssom`): Cross-ontology mappings for disease alignment
- **ICD-10-CM Codes** (`disease_list.raw.icd10_cm_codes`): Official ICD-10 clinical modification codes from CMS
- **MONDO Obsoletion Candidates** (`disease_list.raw.mondo_obsoletion_candidates`): Diseases marked for potential removal from MONDO

## Pipeline Nodes

### Node 1: Ingest Obsoletion Candidates
**Function**: `ingest_obsoletion_candidates`

**Purpose**: Validates and passes through MONDO obsoletion candidates data.

**Inputs**:
- MONDO obsoletion candidates (raw TSV data)

**Outputs**:
- Validated obsoletion candidates dataset (parquet)

**What it does**:
- Validates the schema of obsoletion candidates (mondo_id, label, comment, issue, obsoletion_date)
- Ensures uniqueness of MONDO IDs
- Passes the data through to the primary layer for downstream use

**Why this matters**: Obsoletion candidates should be avoided during curation in Orchard and be filtered out / replaced.

---

### Node 2: Create Billable ICD-10 Codes
**Function**: `create_billable_icd10_codes`

**Purpose**: Identifies which diseases map to billable ICD-10-CM codes and prepares them for injection into MONDO.

**Inputs**:
- ICD-10-CM codes from CMS
- MONDO SSSOM mappings
- ICD-10 prefix and subset parameters

**Outputs**:
- DataFrame with billable ICD-10 to MONDO mappings (subject_id, predicate, object_id)

**What it does**:
1. Formats ICD-10 codes as CURIEs (e.g., "ICD10CM:A00.0")
2. Filters SSSOM mappings to only exact matches (`skos:exactMatch`)
3. Joins ICD-10 codes with MONDO diseases via SSSOM mappings
4. Creates final indicating which MONDO diseases have billable ICD-10 codes

**Why this matters**: Billable ICD-10 codes indicate diseases that are clinically diagnosed and reimbursable, making them high-priority for drug repurposing.

---

### Node 3: Extract Disease Data from MONDO
**Function**: `extract_disease_data_from_mondo`

**Purpose**: The main data extraction engine - queries MONDO ontology to extract all disease information, filters, and metadata.

**Inputs**:
- MONDO graph (RDF/OWL ontology)
- Billable ICD-10 codes (from Node 2)

**Outputs**:
- `mondo_metadata`: Ontology version and metadata
- `mondo_obsoletes`: List of deprecated MONDO terms with replacements
- `disease_list_raw`: Complete disease list with all filter features (32+ boolean columns)
- `mondo_metrics`: Disease hierarchy metrics (currently only descendant counts)
- `subtype_counts`: Aggregated subtype information

**What it does**:

1. **Validates MONDO Graph**: Ensures the ontology was loaded correctly

2. **Extracts Basic Metadata**:
   - Ontology version IRI and title
   - Disease labels and definitions
   - Obsolete terms with replacement mappings

3. **Identifies Disease Subtypes**:
   - Matches diseases against subtype regex patterns
   - Identifies parent disease categories for subtypes
   - Validates subtype-parent relationships using ontology hierarchy
   - Counts subtypes per parent disease
   - Creates subtype subset annotations for injection into MONDO

4. **Extracts Disease Data from Mondo** (via ~40 SPARQL queries):
   - Basic disease information (label, definition, synonyms, cross-references)
   - Metadata annotations (MalaCards linkouts, subsets)
   - 32+ boolean filter features including:
     - **Manual curation filters**: manually included/excluded diseases
     - **Clinical classification**: ClinGen curated, susceptibility diseases
     - **Ontology structure**: subtypes, groupings, leaf nodes
     - **External classification**: Orphanet disorders/subtypes, OMIM entries, ICD-10 codes
     - **Disease categories**: psychiatric, cardiovascular, inflammatory, cancer, paraphilic
     - **Label patterns**: "with or without", "and/or", "acquired"
     - **Hierarchy-based**: unclassified hereditary diseases, grouping ancestors

5. **Computes Hierarchy Metrics**:
   - Counts descendants for each disease (how many child diseases)
   - Used downstream to identify grouping classes vs. specific diseases

**Why this matters**: This node does the heavy lifting of transforming a complex ontology into a structured, queryable dataset with rich features for filtering.

---

### Node 4: Create Disease List
**Function**: `create_disease_list`

**Purpose**: Applies business logic filters, enriches with groupings, and creates the final curated disease list.

**Inputs**:
- Raw disease list with filter features (from Node 3)
- Disease hierarchy metrics (from Node 3)
- Subtype counts (from Node 3)
- Disease list parameters (filter rules, grouping configurations)

**Outputs**:
- Final disease list (`disease_list.prm.disease_list`) with:
  - 32+ boolean filters (renamed from `f_*` to `is_*`)
  - Disease groupings (curated + LLM-generated)
  - Hierarchy metrics
  - Grouping heuristic classification

**What it does**:

1. **Applies MATRIX Inclusion Filters**:
   - **INCLUDE diseases that are**:
     - Manually included (curator override)
     - Leaf nodes (most specific diseases with no children)
     - Direct parents of leaves IF they have OMIM, ICD-10, or Orphanet mappings
     - Mapped to ICD-10 categories
     - Orphanet disorders
     - ClinGen curated
     - OMIM entries

   - **EXCLUDE diseases that are**:
     - Unclassified hereditary diseases (leaf hereditary diseases with no other classification)
     - Paraphilic disorders
     - Manually excluded (curator override)

   - **Conflict Detection**: Raises error if disease is both manually included AND excluded

2. **Renames Filter Columns**: Changes `f_*` prefixes to `is_*` for clarity (e.g., `f_omim` → `is_omim`)

3. **Extracts Disease Groupings**:
   - Parses MONDO subset annotations (semicolon-delimited)
   - Extracts curated groupings (e.g., `harrisons_view`, `mondo_txgnn`, `anatomical`)
   - Extracts LLM-generated groupings
   - Pivots groupings into separate columns with pipe-delimited values

4. **Merges All Data Sources**:
   - Base disease list (filtered data)
   - Disease groupings (pivoted)
   - Hierarchy metrics (descendant counts)
   - Subtype information (subtype counts, parent relationships)

5. **Computes Derived Metrics**:
   - `count_subtypes`: Number of direct subtypes (converted to Int64 nullable)
   - `count_descendants`: Total descendants in hierarchy
   - `other_subsets_count`: Count of other subset memberships (Int64 nullable, preserves pd.NA)
   - `count_descendants_without_subtypes`: Descendants minus subtypes

6. **Applies Grouping Heuristic**:
   - Identifies "grouping" diseases (abstract categories vs. specific diagnoses)
   - Based on subset memberships and descendant counts
   - Uses configurable grouping/non-grouping column rules
   - Overrides for diseases that shouldn't be considered groupings

**Output Schema**:
- ~45 columns including disease metadata, boolean filters, groupings, and metrics
- Validated to have ≥15,000 diseases
- Unique MONDO IDs enforced

**Why this matters**: This creates the final Mondo-derived disease list from which the Orchard disease list is derived.

---

## Output Datasets

### Primary Outputs
- **`disease_list.prm.disease_list`**: Final curated disease list (>22,000 diseases)
- **`disease_list.prm.mondo_metadata`**: MONDO version and provenance information
- **`disease_list.prm.mondo_obsoletes`**: Deprecated diseases with replacement mappings
- **`disease_list.prm.mondo_obsoletion_candidates`**: Diseases marked for potential obsoletion

### Intermediate Outputs
- **`disease_list.int.billable_icd10_codes`**: ICD-10 to MONDO mappings
- **`disease_list.int.disease_list_raw`**: Unfiltered disease data with all features
- **`disease_list.int.mondo_metrics`**: Disease hierarchy metrics
- **`disease_list.int.subtype_counts`**: Subtype relationship data
- **`disease_list.int.mondo_preprocessed`**: Enriched MONDO graph (RDF)

## Key Concepts

### Disease Filters
The pipeline computes 32+ boolean filters that indicate various disease properties. These filters are used to determine disease inclusion/exclusion and provide rich metadata for downstream analysis.

### Disease Subtypes
Some diseases are "subtypes" of more general disease categories (e.g., "Type 1 diabetes" is a subtype of "diabetes mellitus"). The pipeline identifies these relationships using regex patterns and validates them against the ontology hierarchy.

### Disease Groupings
Diseases are organized into multiple grouping systems:
- **Curated groupings**: Hand-curated by domain experts (e.g., Harrison's Principles classifications)
- **LLM groupings**: Generated by language models for additional organization
- **Anatomical groupings**: Based on affected body systems
- **Ontology-derived**: Inherited from MONDO subset annotations

### Grouping Heuristic
A computed flag indicating whether a disease is a "grouping" (abstract category) vs. a specific, diagnosable condition. Groupings typically have:
- Many descendants in the disease hierarchy
- Subset membership indicating they're organizational classes
- Lack of specific clinical mappings (ICD-10, OMIM)

## Data Validation

The pipeline uses Pandera schemas to validate:
- **Input data quality**: Required columns, data types, uniqueness constraints
- **Output data completeness**: Minimum row counts, schema compliance
- **Data integrity**: No duplicate MONDO IDs, all IDs start with "MONDO:"

Validation failures halt the pipeline with descriptive error messages.

## Related Documentation

- [MONDO Disease Ontology](http://obofoundry.org/ontology/mondo.html)
- [SSSOM: Simple Standard for Sharing Ontology Mappings](https://mapping-commons.github.io/sssom/)
