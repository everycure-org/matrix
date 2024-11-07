# Diseases List

The goal of the MATRIX project is to develop strong candidate suggestions for drug repurposing (see [here for a press release](https://everycure.org/every-cure-to-receive-48-3m-from-arpa-h-to-develop-ai-driven-platform-to-revolutionize-future-of-drug-development-and-repurposing/)).

The MATRIX disease list is an effort to construct a list diseases that that can be targeted by drugs.
Many disease terminologies contain diseases that are either disease groupings (such as "hereditary disease" or "cancer") which are too broad to be specifically considered to be targetable by drugs, or diseases that are so specific that they dont have any differentiable criteria relevant to their treatment procedures.
The goal of the MATRIX disease list is, therefore, to separate these cases from "drug-targetable diseases".

The list will be used for communication and navigation purposes

## Maintainer team

| Contributor | Organisation | ORCID |
| ----------- | ------------ | ----- |
| [Melissa Haendel](https://orcid.org/0000-0001-9114-8737)  | Monarch Initiative, Tislab, UNC | [https://orcid.org/0000-0001-9114-8737](https://orcid.org/0000-0001-9114-8737) |
| [Sabrina Toro](https://orcid.org/0000-0002-4142-7153)  | Monarch Initiative, Tislab, UNC | [https://orcid.org/0000-0002-4142-7153](https://orcid.org/0000-0002-4142-7153) |
| [Elliott Sharp](https://everycure.org/leadership-team/elliott-sharp/)  | Every Cure | [https://orcid.org/0000-0003-2955-4640](https://orcid.org/0000-0003-2955-4640) |
| [Nico Matentzoglu](https://orcid.org/0000-0002-7356-1779)  | Monarch Initiative, Independent Consultant | [https://orcid.org/0000-0002-7356-1779](https://orcid.org/0000-0002-7356-1779) |
| [Kevin Schaper](https://orcid.org/0000-0003-3311-7320)  | Monarch Initiative, Tislab, UNC | [https://orcid.org/0000-0003-3311-7320](https://orcid.org/0000-0003-3311-7320) |

## Workflow and Method for creating the MATRIX disease list

This document outlines the basic workflow behind the MATRIX disease list.

1. The disease list corresponds to a subset of the [Mondo disease ontology](https://github.com/monarch-initiative/mondo), which itself can be considered "an ontology of terminologies", integrating other widely used terminologies such as [OMIM](https://omim.org/), [NCIT (for neoplasms)](https://github.com/NCI-Thesaurus/thesaurus-obo-edition), [Orphanet (for Rare Diseases)](https://www.orpha.net/) and [Disease Ontology](https://disease-ontology.org/). It also includes mappings to resources such as [MedGen](https://www.ncbi.nlm.nih.gov/medgen), [UMLS](https://www.nlm.nih.gov/research/umls/index.html) and many others.
1. The list is being kept in sync with the development in Mondo, which includes the addition and removal of disease concepts, synonyms and mappings. This means that if a new disease is added to Mondo, it will be added to the disease list within the same week.
1. The list works my seperating diagnosable, clinically actionable diseases from groupings and theoretical disease subtypes without differentiable diagnostic criteria. This separation works in three steps:
    1. Creating a default designation based on [heuristics](#default-filters-for-disease-list)
    1. Manually curating ambiguous entries according to special prioritisation metrics (essential, we have scores that are indicators of diagnosable diseases, and we we review cases with very low scores)
    1. Keeping the evolving list open to crowd-curation and having the community provide feedback when they come across a missing or wrong entry
1. _Basic workflow_:
    1. Download the latest version of the Mondo disease ontology
    1. Extract all information from Mondo relevant to the disease list as a TSV file, including
        - disease concept metadata such as synonyms and definitions
        - filter criteria such as "grouping" and "subtype" designations
    1. Filter the TSV file according to the [currently agreed heuristics](#default-filters-for-disease-list)
    1. Submit the updated list for review by a member of the [MATRIX disease list core team](#maintainer-team)
    1. Merge and publish the disease list as a versioned artefact on Github in various formats, including TSV and XLSX.


## Default filters for Disease List

Here we outline and motivate the various filters we apply to construct the _diagnosable, clinically actionable diseases_.

As described in our [workflow specification](#workflow-and-method-for-creating-the-matrix-disease-list), these filters serve as _heuristics_ and are gradually overwritten by community and internal expert feedback.

_List of current heurstics_ for the MATRIX disease list:

- INCLUDE all classes that have been designated for inclusion by subject matter experts on the MATRIX project
- INCLUDE all [leaf classes](#leaf-filter)
- INCLUDE all [direct parents of leaf classes](#filter-leaf) IF
    - they [correspond to descendants of OMIM Phenotypic Series (OMIMPS)](#filter-omimps-descendant)
    - they [correspond to Orphanet Subtypes](#filter-orphanet-subtypes)
- INCLUDE all disease classes that [correspond to Orphanet Disorders](#filter-orphanet-disorder)
- INCLUDE all disease classes are [mapped to ICD10 categories](#filter-icd10)
- INCLUDE all disease classes are [curated by ClinGen](#filter-clingen)
- INCLUDE all disease classes are [correspond to OMIM diseases](#filter-omim)
- REMOVE all classes that have been designated for exclusion by subject matter experts on the MATRIX project

<a id="filter-leaf"></a>

### Leaf Filter

##### Heuristic

1. If a disease has no ontological children, it is, by default, included in the list.

##### Background

"Leaf diseases", ie the most specific disease terms in the ontology, most often represent specific diagnosable diseases.
For genetic diseases, these represents diseases caused by variation in a specific gene.

<a id="filter-orphanet-subtypes"></a>

### Orphanet Subtype Filter

##### Heuristic

1. If a disease term in Mondo corresponds to a subtype of a disorder according to [Orphanet](https://www.orpha.net/), it is, by default, included in the list.

##### Background

Orphanet organized their rare disorders as "group of disorders", "disorders", and "subtype of disorders".
They define "subtype of disorders" as sub-forms of a disease based on distinct presentation, etiology, or histological aspect, [see details here](https://www.orpha.net/pdfs/orphacom/cahiers/docs/GB/eproc_disease_inventory_R1_Nom_Dis_EP_04.pdf).
The Mondo disease terms representing diseases considered as "subtype of disorders" in Orphanet are annotated with the 'ordo_subtype_of_a_disorder' subset.
These diseases are most often the most specific disease terms (most often "lead diseases").

<a id="filter-orphanet-disorder"></a>

### Orphanet Disorder Filter

##### Heuristic

1. If a disease term in Mondo corresponds to an Orphanet disorder, it is, by default, included in the list [CHECK!].

##### Background

Orphanet organized their rare disorders as "group of disorders", "disorders", and "subtype of disorders". They define "disorders" as entities including diseases, syndromes, anomalies and particular clinical situations. "Disorders" are clinically homogeneous entities described in at least two independent individuals, confirming that the clinical signs are not associated by fortuity. [REF]
Orphanet conciders this level of classification as "diagnosable" disorder. 
These diseases are most often the ontological parents of "disease subtypes"

<a id="filter-clingen"></a>

### ClinGen Filter

##### Heuristic

1. If a disease is used by the Clinical Genome Resource (ClinGen), it is, by default, included in the list.

##### Background

The Clinical Genome Resource (ClinGen, https://clinicalgenome.org/) is a National Institutes of Health (NIH)-funded resource dedicated to building a central resource that defines the clinical relevance of genes and variants for use in precision medicine and research.

We consider ClinGen diseases/disorders as diagnosable since they are reported in the database and all have directly associated variant information.
ClinGen uses Mondo directly during curation, see for example [https://search.clinicalgenome.org/kb/conditions/MONDO:0020119](https://search.clinicalgenome.org/kb/conditions/MONDO:0020119).

<a id="filter-omim"></a>

### OMIM Filter

##### Heuristic

1. If a disease has an exact match to an OMIM identifier (ie disease entry), it is, by default, included in the list.

##### Background

The Online Mendelian Inheritance in Man (OMIM, https://www.omim.org/) catalogs human genes and genetic disorders and traits.
All OMIM genetic disorders have direct, equivalent correspondences in Mondo.
We consider OMIM diseases/disorders as diagnosable (since they are reported in the database).

<a id="filter-icd10"></a>

### ICD10 CM Filter

##### Heuristic

1. If a disease has an exact match to an ICD 10 category code, it is, by default, included in the list.
1. If a disease has an exact match to an ICD 10 chapter or chapter header code, it is, by default, excluded from the list.

##### Background

There are a few different types of ICD-10 codes that can be roughly identified by their structure:

1. Chapter codes (or block codes), for example `A00-B99` (Certain infectious and parasitic diseases). These codes can be recognised by containing a dash (`-`) character.
1. Chapter headers (or chapter titles), for example `A00` (Cholera). These can be identified by neither containing a dash, nor a period (`.`) character.
1. Category codes (or subcategory codes), for example: `A01.1` (Paratyphoid fever A). These can be recognized by containing a period (`.`) character.

Usually, we can assume the following:

1. The codes with dashes (chapter codes) represent broad categories of diseases.
1. The codes with periods (category/subcategory codes) represent more specific diagnoses.
1. The codes without dashes or periods (chapter headers) are usually the top-level categories within each chapter.

In clinical and coding contexts, people often refer to the codes with periods as the "billable codes" or "billable ICD-10 codes" because these are typically the ones used for specific diagnoses in medical billing and record-keeping. Codes without a period (chapter headers) are generally not billable, and Codes with dashes (chapter codes/block codes) are never billable.

However, it's important to note that not all codes with periods are billable. Some may require additional digits for specificity. The exact rules can vary slightly depending on the specific implementation of ICD-10 (such as ICD-10-CM in the United States), but generally, the most specific codes (usually those with periods) are the billable ones.

<a id="filter-omimps"></a>

### OMIMPS Filter

##### Heuristic

1. If a disease has an exact match to an OMIMPS identifier, it is, by default, excluded from the list.

##### Background

OMIM Phenotypic Series (OMIMPS) group diseases based on similar phenotypes.
OMIMPS most often refers to the general disease when the OMIM terms are gene-specific subtypes of the disease. For example, the OMIMPS ["Usher syndrome"](https://www.omim.org/phenotypicSeries/PS276900) includes all subtypes of Usher syndrome.
Sometimes, the OMIMPS group terms based on phenotype similarities, for example ["Intellectual developmental disorder, X-linked syndromic"](https://www.omim.org/phenotypicSeries/PS309510).
By nature, Mondo terms representing OMIMPS entry are not actual diseases but group of diseases.

<a id="filter-omimps-descendant"></a>

### OMIMPS descendant Filter

##### Heuristic

1. If a disease is a subclass of a disease that corresponds to an OMIM Phenotypic Series, it is, by default, included in the list.

##### Background

Since OMIMPS group diseases, we determined that ontological children of OMIMPS should be diseases that we would want to include.
These would include terms corresponding to OMIM terms, and possibly other disease terms.

<a id="filter-grouping-subset"></a>

### Grouping subset Filter

##### Heuristic

1. If a disease term in Mondo corresponds to a _group of disorders_ according to Orphanet (has a 'ordo_group_of_disorders' subset annotation), it is, by default, excluded in the list
1. If a disease term in Mondo has a 'disease_grouping' subset annotation, it is, by default, excluded in the list
1. If a disease term in Mondo has a 'harrisons_view' subset annotation, it is, by default, excluded in the list
1. If a disease term in Mondo has a 'rare_grouping' subset annotation, it is, by default, excluded from the list

##### Background

By nature, Mondo terms in the following subsets are not actual diseases but group of diseases.
- _'ordo_group_of_disorders' subset_: Orphanet organized their rare disorders as "group of disorders", "disorders", and "subtype of disorders". They define "group of disorders" as a collection of disease/clinical entities sharing a given characteristic. [REF]
- _'disease_grouping' subset_: Terms in this subset have been manually curated and determined to be a grouping term.
- _'harrisons_view' subset_: Mondo's high-level classification was created based on the Harrison’s Principle of Internal Medicine textbook. Terms representing this high-level classification are annotated with the 'harrisons_view' subset
- _'rare_grouping' subset_: The ontological parent of rare diseases (see Mondo rare disease subset [here](https://mondo.readthedocs.io/en/latest/editors-guide/rare-disease-subset/))

<a id="filter-grouping-subset-ancestor"></a>

### Grouping Subset Ancestor Filter

##### Heuristic

1. If a disease is an ontological parent of a disease that is a grouping term (as defined in the "Grouping Subset Filter" section), it is, by default, excluded from the list.

##### Background

Ontologically, a parent of a grouping class would itself be a grouping class.

<a id="filter-direct-parent"></a>

### Leaf Direct Parent Filter

##### Heuristic

1. This filter indicates if a disease is a direct parent of a leaf term
1. This filter is for information purposes and is not used to include/exclude terms from the list [NOT TRUE!].

##### Background

This filter exists for information purposes. We think that the majority of the "leaf direct parent" would also be in the "orphanet disorder" subset and in the "OMIM" subset, and therefore whould be included in the list.

<a id="filter-subtype-descendant"></a>

### Subtype Subset Descendant Filter

##### Heuristic

1. This filter indicates if a disease is an ontological child of a "subtype of disorders" term.
1. This filter is for information purposes and is not used to include/exclude terms from the list.

##### Background

This filter exists for information purposes. We think that the majority of the "orphanet subtype of disorder" would be leaf terms as they are specific. However, if there is a term that is an ontological child of an "orphanet subtype of disorder", it might need to be be included in the list .

### Limitations of the filtering approach created in this section:

- Mondo mappings to ICD10CM are currently incomplete, therefore the filter will result in false positives and false negatives.