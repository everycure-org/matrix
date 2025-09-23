# Attribution

!!! tip "General acknowledgement"

    Work in the Matrix project was made possible by the contributions of countless free-software developers and open data curators, upon which much of our modern infrastructure is built.

In the following we acknowledge some of the core resources that drive our success.

- **Data sources**
    - [First level knowledge providers](#knowledgeproviders) are the immediate data sources included in the Matrix project, typically for the purpose of pair prediction.
        - [RTX-KG2](#rtxkg2)
        - [ROBOKOP](#robokop)
        - [SPOKE KG](#spoke)
        - [Embiology](#embiology)
        - [PrimeKG](#primekg)
    - [Primary knowledge sources](primary_knowledge_sources.md) are the raw data sources that are part of the knowledge graphs used for pair prediction.
    - [Ground Truth lists](../pipeline/data/ground_truth_lists.md) serve as evaluation data for drug-disease pair prediction algorithms.
    - [Mondo Disease Ontology](#mondo) is used as the backbone for the disease list.
- **Software sources**
    - [Kedro](#kedro)
    - [PySpark](#pyspark)
    - [Docker](#docker)
    - [Neo4J](#neo4j)
    - [MLFlow](#mlflow)
    - [Kubernetes (K8s)](#kubernetes)
    - [Argo Workflows](#argoworkflows)
    - [Terraform](#terraform)
    - [Docker compose](#docker-compose)
    - [NCATS Node Normalizer](#ncats-nn)
    - [NCATS Node Resolver](#ncats-nr)
    - [ARAX Node Normalizer](#arax-nn)

## Data sources

<a id="knowledgeproviders"></a>

### First-level data providers

First-level data providers are those that we leverage directly in the context of the Matrix pipeline.

<a id="rtxkg2"></a>

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

<a id="robokop"></a>

#### ROBOKOP

MATRIX builds on resources from ROBOKOP (Reasoning Over Biomedical Objects linked in Knowledge Oriented Pathways), a question-answering system and knowledge graph developed as part of the NCATS Translator program.
ROBOKOP combines graph reasoning services with biomedical knowledge integration, enabling exploration of mechanistic hypotheses across diseases, drugs, and biological processes.
Its graph-based reasoning services have informed MATRIX’s approach to query expansion, pathway exploration, and candidate prioritization.
For details, see the [ROBOKOP portal](https://robokop.renci.org/) and [this publication](https://pubmed.ncbi.nlm.nih.gov/31769676/):

```
Bizon C, Cox S, Balhoff J, Kebede Y, Wang P, Morton K, Fecho K, Tropsha A. ROBOKOP KG and KGB: Integrated Knowledge Graphs from Federated Sources. J Chem Inf Model. 2019 Dec 23;59(12):4968-4973. doi: 10.1021/acs.jcim.9b00683. Epub 2019 Dec 12. PMID: 31769676; PMCID: PMC11646564.
```

<a id="spoke"></a>

#### SPOKE

!!! warning "Private data source"

    Note that Every Cure utilize this data source in the MATRIX pipeline but do not distribute it, please reach out to data owners directly for access.


MATRIX builds on SPOKE (Scalable Precision Medicine Oriented Knowledge Engine), a large heterogeneous biomedical knowledge graph developed at UCSF.
SPOKE integrates a wide variety of biomedical databases into a single graph, capturing relationships among genes, proteins, diseases, drugs, and clinical concepts.
Its graph-based representations have informed MATRIX’s downstream analyses for identifying novel therapeutic opportunities.
For details, see the [SPOKE portal](https://spoke.ucsf.edu/) and [this publication](https://doi.org/10.1093/bioinformatics/btad080):

```
Morris JH, Soman K, Akbas RE, Zhou X, Smith B, Meng EC, Huang CC, Cerono G, Schenk G, Rizk-Jackson A, Harroud A, Sanders L, Costes SV, Bharat K, Chakraborty A, Pico AR, Mardirossian T, Keiser M, Tang A, Hardi J, Shi Y, Musen M, Israni S, Huang S, Rose PW, Nelson CA, Baranzini SE. The scalable precision medicine open knowledge engine (SPOKE): a massive knowledge graph of biomedical information. Bioinformatics. 2023 Feb 3;39(2):btad080. doi: 10.1093/bioinformatics/btad080. PMID: 36759942; PMCID: PMC9940622.
```

<a id="embiology"></a>

### Embiology

!!! warning "Private data source"

    Note that Every Cure utilize this data source in the MATRIX pipeline but do not distribute it, please reach out to data owners directly for access.

MATRIX leverages Embiology, a proprietary dataset from Elsevier that encodes curated relationships among biomedical entities extracted from the scientific literature.
Embiology combines large-scale natural language processing with expert curation to capture connections between diseases, drugs, targets, and mechanisms of action to 
better understand disease biology.
[Further information on Embiology](https://www.elsevier.com/products/embiology) is available from Elsevier.

<a id="primekg"></a>

### PrimeKG

MATRIX builds on resources from PrimeKG, a precision medicine knowledge graph developed to support drug repurposing and clinical translation research.
PrimeKG integrates a wide range of biomedical entities — including diseases, drugs, genes, and biological pathways — into a single harmonized framework.
For details, see the [GitHub repo](https://github.com/mims-harvard/PrimeKG) and [this publication](https://pubmed.ncbi.nlm.nih.gov/31769676/):

```
Chandak, P., Huang, K. & Zitnik, M. Building a knowledge graph to enable precision medicine. Sci Data 10, 67 (2023). https://doi.org/10.1038/s41597-023-01960-3  
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
This list is not exhaustive.
If you think a piece of software is worth highlighting here, let us know
on our [issue tracker](https://github.com/everycure-org/matrix/issues).

<a id="kedro"></a>

### [Kedro](https://kedro.org/)  

A Python framework for building reproducible, maintainable data science pipelines, used in MATRIX to structure and orchestrate ETL workflows.  

<a id="pyspark"></a>

### [PySpark](https://spark.apache.org/docs/latest/api/python/)  

The Python interface to Apache Spark, enabling distributed data processing and large-scale transformations in MATRIX’s integration pipeline.  

<a id="docker"></a>

### [Docker](https://www.docker.com/)  

A containerization platform that ensures MATRIX software components run in consistent, portable environments.  

<a id="neo4j"></a>

### [Neo4j](https://neo4j.com/)  

A graph database optimized for querying and exploring biomedical relationships, used to host and analyze the integrated MATRIX knowledge graph.  

<a id="mlflow"></a>

### [MLflow](https://mlflow.org/)  

An open-source platform for managing machine learning experiments, tracking, and reproducibility within MATRIX’s AI workflows.  

<a id="kubernetes"></a>

### [Kubernetes (K8s)](https://kubernetes.io/)  

A container orchestration system that manages scaling, deployment, and resilience of MATRIX’s cloud-native components.  

<a id="argoworkflows"></a>

### [Argo Workflows](https://argoproj.github.io/workflows/)  

A Kubernetes-native workflow engine used for defining and running MATRIX’s complex, multi-step data pipelines.  

<a id="terraform"></a>

### [Terraform](https://www.terraform.io/) / [Terragrunt](https://terragrunt.gruntwork.io/)  

Infrastructure-as-code tools that provision and manage MATRIX’s cloud environments in a reproducible and versioned way.  

<a id="docker-compose"></a>

### [Docker Compose](https://docs.docker.com/compose/)  

A tool for defining and running multi-container applications locally, supporting MATRIX development and testing.  

<a id="ncats-nn"></a>

### [NCATS Node Normalizer](https://nodenormalization.transltr.io/)  

A Translator service for mapping biomedical entity identifiers across vocabularies, supporting consistent normalization in MATRIX.  

<a id="ncats-nr"></a>

### [NCATS Name Resolver](https://name-resolution-sri.renci.org/)  

A Translator service for resolving biomedical entity names into standardized identifiers used throughout MATRIX.  

<a id="arax-nn"></a>

### [ARAX Node Normalizer](https://arax.rtx.ai/)  

An alternative node normalization service developed by the ARAX team, leveraged in MATRIX for identifier harmonization and redundancy checks.  

---
