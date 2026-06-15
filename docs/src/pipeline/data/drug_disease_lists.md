# Diseases List

The goal of the MATRIX project is to develop strong candidate suggestions for drug repurposing (see [here for a press release](https://everycure.org/every-cure-to-receive-48-3m-from-arpa-h-to-develop-ai-driven-platform-to-revolutionize-future-of-drug-development-and-repurposing/)).

The MATRIX disease list is an effort to construct a list diseases that that can be targeted by drugs.
Many disease terminologies contain diseases that are either disease groupings (such as "hereditary disease" or "cancer") which are too broad to be specifically considered to be targetable by drugs, or diseases that are so specific that they dont have any differentiable criteria relevant to their treatment procedures.
The goal of the MATRIX disease list is, therefore, to separate these cases from "drug-targetable diseases" with help of external ontologies and manual curation.

The list will be used for communication and navigation purposes

## Availability on HuggingFace Hub

The EC Disease List is available on HuggingFace Hub at [everycure/disease-list](https://huggingface.co/datasets/everycure/disease-list) under the CC-BY-4.0 license. The HuggingFace dataset is automatically updated with each minor and major release (patch releases are not included).

The public HuggingFace version is designed to be clear, stable, and broadly usable. As part of that goal, certain internal-use fields that are still evolving are not included in the public release.

EC team members looking for the full dataset, including all internal fields, should refer to the datasets repository.

## Workflow and Method for creating the MATRIX disease list

Details for how MATRIX disease list is created can be found within the [core entities pipeline project](../../../../pipelines/core_entities/src/core_entities/pipelines/disease_list/).
In short, MONDO disease ontology is used to extract diseases which are then enriched with curated annotations, producing the final MATRIX disease list availalbe on Huggin Face.

## Disease list subsets / tagging / grouping

We have developed a system for grouping diseases into categories (called here tagging or grouping).

This has various purposes:

1. _Model evaluation_. Similar to the approach of the TxGNN paper, we need to be able to evaluate whether a model trained for one disease area will also work for another. In other words, we do the train - test split by disease "area" (say, endocrine disease for training and cancer for testing).
2. _Display_. Sometimes, we want to be able to offer sensible facets in a search interface to a user, like "endocrine system diseaes".

There are three major ways we add these groupings:

1. Mondo subset tags. Mondo itself curates certain subsets, such as the "Harrisons view", which corresponds to the disease categories from a [popular medical textbook](https://accessmedicine.mhmedical.com/book.aspx?bookID=3095).
2. LLM-generated subset tags. We have developed a [Kedro pipeline](https://github.com/everycure-org/matrix/tree/main/pipelines/core_entities) which generates disease tags flexibly using LLMs.
3. Manual curation of subset grouping classes. We have provided a way to manually curate subset groupings, so that new subsets can be defined as needed.

### Mondo subset tags

(Note this list may be out of date)

Mondo provides two majore subsets:

1. Its manually curated class hierarchy. Everything under the [human disease](https://www.ebi.ac.uk/ols4/ontologies/mondo/classes/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FMONDO_0700096?lang=en) concept is considered a subset. The advantage of this subset is that it will always cover the entirety of all human diseases in Mondo.
2. The Harrison subset. Corresponding to a [popular medical textbook](https://accessmedicine.mhmedical.com/book.aspx?bookID=3095), this subset contains most Mondo classes, using the groupings provided by the textbook.
Note, at the time of this writing the Harrison view and the manually curated disease hierarchy are the same (this was not the case, for example last month, and might change again in the future.)

### Manually curated subset tags

We have developed a template based approach [using a spreadsheet to manually specify subsets](https://github.com/everycure-org/matrix-disease-list/blob/main/src/grouping-diseases.robot.tsv).

The curator simply specifies the root nodes of the Mondo ontology to include in any given subset, say "endocrine disorder" and "cancer".

The pipeline than creates tags based on those groups that are then exported into the disease grouping list.

### Mannual annotations of clinical recognizability

As not all entities within a MONDO ontology will correspond to diseases relevant to drug repurposing and recognized by medical community, we manually annotated all MONDO diseases based on their clinical recognizability. 

To each disease, we assigned a `level` attribute which describes whether the ontological entity describes a clinically recognized disease, its subgroup or a high-level grouping.

* Groupings were assigned when entity was representing a high-level disease grouping which cannot be easily targettable by drugs, such as [neurodegenerative disease](https://www.ebi.ac.uk/ols4/ontologies/mondo/classes/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FMONDO_0005559?lang=en) 
* Clinically recognized label was assigned when an entity was representing a disease which would be recognized by both a physician and a patient diagnosed with the disease, such as [Alzheimer disease](https://www.ebi.ac.uk/ols4/ontologies/mondo/classes/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FMONDO_0004975?lang=en) 
* Subgroups were assigned for clinically identical subtypes of diseases represented by different leaf nodes (e.g. due to varying chromosome mutations, such as in the case of [Alzheimer 17](https://www.ebi.ac.uk/ols4/ontologies/mondo/classes/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FMONDO_0014036?lang=en) and [Alzheimer 18](https://www.ebi.ac.uk/ols4/ontologies/mondo/classes/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FMONDO_0014265?lang=en)).

Note that the motivation for those groupings is to try to capture what physicians or patients would recognize. We acknowledge that disease definition is a very complex issue and this classification is only 
present to help navigate ontologies rather than replace any existing guidelines.

#### Outputs

[A table](https://raw.githubusercontent.com/everycure-org/matrix-disease-list/refs/heads/main/matrix-disease-groupings.tsv) is generated that looks like this:

| category_class | label                                         | harrisons_view            | matrix_txgnn_grouping     | mondo_top_grouping                                                     |
|----------------|-----------------------------------------------|---------------------------|---------------------------|------------------------------------------------------------------------|
| MONDO:0000004  | adrenocortical insufficiency                  | endocrine_system_disorder | endocrine_system_disorder | endocrine_system_disorder                                              |
| MONDO:0000005  | alopecia, isolated                            | hereditary_disease        | other                     | integumentary_system_disorder|hereditary_disease                       |
| MONDO:0000009  | inherited bleeding disorder, platelet-type    | hereditary_disease        | other                     | hematologic_disorder|hereditary_disease                                |
| MONDO:0000014  | colorblindness, partial                       | other                     | psychiatric_disorder      | nervous_system_disorder|psychiatric_disorder|disorder_of_visual_system |
| MONDO:0000015  | classic complement early component deficiency | hereditary_disease        | other                     | hereditary_disease|immune_system_disorder  

The first column (`category_class`) is the tagged disease in Mondo, and the second column (`label`) the corresponding disease name.

All the remaining columns corresponds to different groupings. In the example we show three groupings:

1. `harrisons_view`
1. `matrix_txgnn_grouping`
1. `mondo_top_grouping`

The values are the tags, so which specific disease group a disease corresponds to. For example: "adrenocortical insufficiency" (`MONDO:0000004`) corresponds to an `endocrine_system_disorder` according to the `harrisons_view`.
