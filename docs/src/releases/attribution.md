# Attribution

!!! tip "General acknowledgement"

    Work in the Matrix project was made possible by the contributions of countless free-software developers and open data curators, upon which much of our modern infrastructure is built.

- **Data sources**
    - [First level knowledge providers](#knowledgeproviders) are the immediate data sources included in the Matrix project, typically for the purpose of pair prediction.
    - [Primary knowledge sources](primary_knowledge_sources.md) are the raw data sources that are part of the knowledge graphs used for pair prediction.
    - [Ground Truth lists](../pipeline/data/ground_truth_lists.md) serve as evaluation data for drug-disease pair prediction algorithms.
    - [Mondo Disease Ontology](#mondo) is used as the backbone for the disease list.
- **Software sources**
    - [Kedro](#kedro) is a toolbox for production-ready data science and underlies all Matrix pipelines.

## Data sources

<a id="knowledgeproviders"></a>

### First level data providers

#### RTX-KG2

MATRIX integrates information from RTX-KG2, a large-scale biomedical knowledge graph developed by the Translator RTX team.
RTX-KG2 aggregates and harmonizes knowledge from dozens of authoritative biomedical databases and ontologies into a single, semantically consistent graph aligned with the Biolink Model.
It provides a rich source of curated biological and clinical associations that support reasoning and drug repurposing use cases.
For details on sources and construction, see RTX-KG2 documentation and [this publication](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-022-04932-3):

```
Wood, E.C., Glen, A.K., Kvarfordt, L.G. et al. 
RTX-KG2: a system for building a semantically standardized knowledge graph for translational biomedicine.
BMC Bioinformatics 23, 400 (2022).
doi: 10.1186/s12859-022-04932-3
```

#### ROBOKOP

MATRIX builds on resources from ROBOKOP (Reasoning Over Biomedical Objects linked in Knowledge Oriented Pathways), a question-answering system and knowledge graph developed as part of the NCATS Translator program.
ROBOKOP combines graph reasoning services with biomedical knowledge integration, enabling exploration of mechanistic hypotheses across diseases, drugs, and biological processes.
Its graph-based reasoning services have informed MATRIX’s approach to query expansion, pathway exploration, and candidate prioritization.
For details, see the [ROBOKOP portal](https://robokop.renci.org/) and [this publication](https://pubmed.ncbi.nlm.nih.gov/31769676/):

```
Bizon C, Cox S, Balhoff J, Kebede Y, Wang P, Morton K, Fecho K, Tropsha A.
ROBOKOP KG and KGB: integrated knowledge graphs from federated sources. 
J Chem Inf Model 2019 Dec 23;59(12):4968–4973. 
doi: 10.1021/acs.jcim.9b00683
```

#### SPOKE

MATRIX builds on SPOKE (Scalable Precision Medicine Oriented Knowledge Engine), a large heterogeneous biomedical knowledge graph developed at UCSF.
SPOKE integrates a wide variety of biomedical databases into a single graph, capturing relationships among genes, proteins, diseases, drugs, and clinical concepts.
Its graph-based representations have informed MATRIX’s downstream analyses for identifying novel therapeutic opportunities.
For details, see the [SPOKE portal](https://spoke.ucsf.edu/) and this publication:

```
Nelson CA, Butte AJ, Bonham VL, Ji HP, Guo X, Chang S, Lancia S, Fong R, Ganapathiraju MK, Greene CS, Himmelstein DS, Maayan A, McInnes L, Tatonetti NP, Wong AK, Zheng NS, Greene D, Berkovic SF, Brennan P, Cozen W, Morton LM, Zhang Y, Chanock SJ, Yu B, Bult CJ, Haendel M, Staudt LM, Crawford DC, Pendergrass SA, Bush WS.  
A large-scale biomedical knowledge graph for precision medicine.  
Nat Commun. 2017 Nov 29;8(1):15014.  
doi: 10.1038/ncomms15014
```

<a id="mondo"></a>

### Mondo Disease Ontology

MATRIX builds on resources from Mondo Disease Ontology (MONDO), an open, community-driven ontology that harmonizes disease definitions across numerous medical vocabularies.
MONDO provides a unified, semantically consistent set of disease identifiers, enabling interoperability across biomedical datasets and facilitating disease-centric reasoning within MATRIX.
Its integrative approach to aligning rare and common diseases is especially valuable for drug repurposing applications.
For details, see the [Mondo project page](https://mondo.monarchinitiative.org/) and [this publication](https://www.medrxiv.org/content/10.1101/2022.04.13.22273750v3).

```
Nicole A Vasilevsky et. al
Mondo: Unifying diseases for the world, by the world.
medRxiv 2022.04.13.22273750
doi: doi:10.1101/2022.04.13.22273750
```

## Software

Here we acknowledge a few of the central pieces in our ecosystem.

<a id="kedro"></a>

### Kedro

https://kedro.org/