## Workflow and Method for creating the MATRIX disease list

This document outlines the basic workflow behind the MATRIX disease list.

1. The disease list corresponds to a subset of the [Mondo disease ontology](https://github.com/monarch-initiative/mondo), which itself can be considered "an ontology of terminologies", integrating other widely used terminologies such as [OMIM](https://omim.org/), [NCIT (for neoplasms)](https://github.com/NCI-Thesaurus/thesaurus-obo-edition), [Orphanet (for Rare Diseases)](https://www.orpha.net/) and [Disease Ontology](https://disease-ontology.org/). It also includes mappings to resources such as [MedGen](https://www.ncbi.nlm.nih.gov/medgen), [UMLS](https://www.nlm.nih.gov/research/umls/index.html) and many others.
1. The list is being kept in sync with the development in Mondo, which includes the addition and removal of disease concepts, synonyms and mappings. This means that if a new disease is added to Mondo, it will be added to the disease list within the same week.
1. The list works my seperating diagnosable, clinically actionable diseases from groupings and theoretical disease subtypes without differentiable diagnostic criteria. This separation works in three steps:
    1. Creating a default designation based on [heuristics](filtering.md)
    1. Manually curating ambiguous entries according to special prioritisation metrics (essential, we have scores that are indicators of diagnosable diseases, and we we review cases with very low scores)
    1. Keeping the evolving list open to crowd-curation and having the community provide feedback when they come across a missing or wrong entry
1. _Basic workflow_:
    1. Download the latest version of the Mondo disease ontology
    1. Extract all information from Mondo relevant to the disease list as a TSV file, including
        - disease concept metadata such as synonyms and definitions
        - filter criteria such as "grouping" and "subtype" designations
    1. Filter the TSV file according to the [currently agreed heuristics](filtering.md)
    1. Submit the updated list for review by a member of the [MATRIX disease list core team](index.md)
    1. Merge and publish the disease list as a versioned artefact on Github in various formats, including TSV and XLSX.

