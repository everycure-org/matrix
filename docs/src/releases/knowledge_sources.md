# First-level Knowledge Sources

First-level knowledge sources are those that we leverage directly in the context of the Matrix pipeline.

- [RTX-KG2](#rtxkg2)
- [ROBOKOP](#robokop)
- [SPOKE KG](#spoke)
- [EmBiology](#embiology)
- [PrimeKG](#primekg)

------------------

<a id="robokop"></a>

### ROBOKOP (Reasoning Over Biomedical Objects Linked in Knowledge Oriented Pathways)

- **Homepage**: [https://robokop.renci.org/](https://robokop.renci.org/)
- **Graph Version/ Release:** See [Matrix release pages](https://docs.dev.everycure.org/releases/release_history/)
- **Current status:** Active, publicly accessible UI and KG endpoints; open‑source project

#### Rationale for inclusion in EC KG

Open, Translator-aligned biomedical KG with a mature question‑answering interface and rich multi‑source integration.
Demonstrated utility for hypothesis generation and drug repurposing tasks (e.g., COVID‑related efforts).
Suitable for transparent provenance and iterative curation.

#### Citations

* **Primary**: `Bizon C. et al. ROBOKOP KG & KGB: integrated knowledge graphs from federated sources. J Chem Inf Model. 2019`, [PMID:31549975]().
* `Morton K. et al. ROBOKOP: abstraction layer & UI for knowledge graph–based question answering. Bioinformatics. 2019`, [PMID:31077396]().
* `Korn D. et al. COVID-KOP: Integrating Emerging COVID-19 Data with the ROBOKOP Database. Bioinformatics. 2021;37(4):586-587`, [PMID:33175089]().

#### Drug Repurposing-related Use Cases

* Used to refine mechanism-of-action (MOA) hypotheses, resulting in more targeted identification of candidate drugs ([PMID:XXX]()).  
* Applied to virtual screening workflows, resulting in prioritized drug–target associations for repurposing exploration ([PMID:31549975]()).  
* Used to generate hypotheses for drug repurposing of known drugs and clinical candidates against COVID-19, resulting in mechanistic pathway suggestions ([PMID:33175089]()).

#### Funding / Supporting Programs

* NIH NCATS ([prototype: **OT2TR002514**](https://robokop.renci.org/#funding))
* Ongoing support: NIH **NIEHS** & **Office of Data Science Strategy (ODSS)** ([**U24ES035214**](https://robokop.renci.org/#funding))
* [RENCI (UNC‑Chapel Hill) & CoVar LLC collaboration](https://robokop.renci.org/#funding)

#### Licensing & Accessibility

* **License**: [MIT (software and KG)](https://robokop.renci.org/question-builder/about)
* **Accessibility**: Public UI and [APIs for exploratory queries](https://robokop.renci.org/api-docs/docs/automat/robokop-kg); programmatic access available
* **Note on reuse**: Upstream source licenses apply; retain original terms when redistributing

------------------

<a id="rtxkg2"></a>

### RTX‑KG2 (for the ARAX reasoning system)

- **Homepage**: [https://github.com/RTXteam/RTX-KG2](https://github.com/RTXteam/RTX-KG2)
- **Graph Version/ Release:** See [Matrix release pages](https://docs.dev.everycure.org/releases/release_history/)
- **Current status:** Active, publicly accessible through Translator system; open‑source project

#### Rationale for inclusion in EC KG

Large, semantically standardized translational KG integrating UMLS, SemMedDB, ChEMBL, DrugBank, 
Reactome and many more sources. Strong coverage for Drug–Target–Disease reasoning and compatibility with 
downstream reasoning agents.

#### Citations

* **Primary**: `Wood E. C. et al. RTX-KG2: a system for building a semantically standardized knowledge graph for translational biomedicine. BMC Bioinformatics. 2022`, [PMID:35423592]().
* `Ma C. et al. KGML-xDTD: a knowledge graph-based machine learning framework for drug treatment prediction and mechanism description. (leveraging RTX-KG2 canonical graph) GigaScience / PMC / preprint. 2023`, [PMID:37228255]().
* `Glen A.K.  et al. ARAX: a graph-based modular reasoning tool for translational biomedicine. Bioinformatics. 2023 Mar;39(3):btad082`, [PMID:36773175]().

#### Drug Repurposing-related Use Cases

* Used to train KGML-xDTD, which predicts drug–disease treatment probabilities and also provides explainable mechanism paths via RTX-KG2c; in comparisons, showed higher accuracy and fewer false positives in drug repurposing tasks ([PMID:37228255]()).  
* Used as the foundational knowledge graph in the Translator system’s reasoning agents (ARAX, mediKanren, etc.) to generate drug repurposing hypotheses by linking drugs, targets, and diseases via the rich source integration in RTX-KG2 ([PMID:35423592]()).
* ARAX’s “Recovering drug/disease relationships” use case: ARAX, which uses RTX-KG2 as a knowledge provider, has explicit workflows/query-types that recover known drug/disease pairs, which helps validate repurposing pipelines ([PMID:36773175]()).

#### Funding / Supporting Programs

* NIH NCATS Translator program ([award **OT2TR002520**](https://github.com/RTXteam/RTX-KG2))
* NIH NCATS funded project ([**3OT2TR003428-01S1**](https://reporter.nih.gov/search/rgDgbiaqU0a6v1T6SU-4xA/project-details/10333468#details))

#### Licensing & Accessibility

* **License**: Build and Software provided free-of-charge under the MIT License; documentation & downloadable build artifacts distributed under CC-BY 4.0
* **Accessibility**: Public GitHub repo, S3 buckets, and APIs available; programmatic access available
* **Note on reuse**: Knowledge graph content inherits licenses from upstream sources; downstream users must comply with original source licenses when reusing KG2 content

------------------

<a id="spoke"></a>

### SPOKE (Scalable Precision Medicine Oriented Knowledge Engine)

- **Homepage**: [https://spoke.rbvi.ucsf.edu/](https://spoke.rbvi.ucsf.edu/)
- **Current status:** Continuously maintained academic/enterprise offering with public web explorer for neighborhoods; full graph access under license

#### Rationale for inclusion in EC KG

Precision medicine–focused KG that links dozens of biomedical databases into a unified network.
Demonstrated predictive signal for indications and translational analyses;
aligns well with clinical and molecular use cases in repurposing.

#### Citations

* **Primary**: `Morris J.H. et al. The scalable precision medicine open knowledge engine (SPOKE): a massive knowledge graph of biomedical information. Bioinformatics. 2023 Feb;39(2):btad080`, [PMID:36759942]().
* `Baranzini S.E. et al. A Biomedical Open Knowledge Network Harnesses the Power of AI to Understand Deep Human Biology. AI Magazine. 2022 Mar;43(1):46-58`, [PMID:36093122]().
* `Soman K., et al. Biomedical knowledge graph-optimized prompt generation for large language models. Bioinformatics. 2024 Sep;40(9)`, [PMID:39288310]().

#### Drug Repurposing-related Use Cases

* Used to explore drug-disease neighborhoods via the Neighborhood Explorer; enabling hypothesis generation for repurposing, including exploring connections of SARS-CoV-2 spike protein to human proteins and compounds for potential interventions. (PMID:36759942).  
* Used to uncover possible mechanistic pathways linking ACE2 upregulation through mechanical ventilation and its modulation by dexamethasone via the SPOKE graph; this generated a drug repurposing candidate in the COVID-19 context. (From *A Biomedical Open Knowledge Network Harnesses …*) (PMID:36093122).
* Used to optimize prompt generation for large language models by incorporating SPOKE as the retrieval knowledge base; this helps with generating evidence-backed drug/disease associations in prompts, which may assist repurposing research workflows. (PMID:39288310\)

#### Funding / Supporting Programs

* National Science Foundation ([**NSF_2033569**](https://academic.oup.com/bioinformatics/article/39/2/btad080/7033465))
* NIH/NCATS ([**NIH_NOA_1OT2TR003450**](https://academic.oup.com/bioinformatics/article/39/2/btad080/7033465))
* [Marcus Program in Precision Medicine Innovation](https://academic.oup.com/bioinformatics/article/39/2/btad080/7033465)

#### Licensing & Accessibility

* **License**: [Build software provided free-of-charge under the MIT License](https://spoke.rbvi.ucsf.edu/license.html); documentation & downloadable build artifacts distributed under CC-BY 4.0
* **Accessibility**: Public GitHub repo, S3 buckets, and APIs available; downstream users must comply with original source licenses when reusing content
* **Note on reuse**: Knowledge graph content inherits licenses from upstream sources; our CC-BY assertion applies only to creative products we generate

------------------

<a id="embiology"></a>

### EmBiology (Elsevier Biology Knowledge Graph)

- **Homepage**: [https://www.elsevier.com/products/embiology](https://www.elsevier.com/products/embiology)
- **Current status:** Commercial, continuously updated product; vendor reports frequent KG refreshes (weekly to annual components)

#### Rationale for inclusion in EC KG

Curated, AI‑driven knowledge graph spanning literature, trials, and databases with strong cause‑effect and biological relation coverage; 
useful for repurposing hypothesis generation, target/biomarker discovery, and evidence triage.

#### Citations

* [Elsevier launches EmBiology](https://www.elsevier.com/about/press-releases/elsevier-launches-embiology-to-deliver-the-unparalleled-insights-into-biological-activities-that-accelerate-drug-discovery?) to deliver unparalleled insights into biological activities that accelerate drug discovery. Press release. April 19, 2023.
* EmBiology | Biological data structured for insights. [Product page](https://www.elsevier.com/products/embiology?).

#### Drug Repurposing-related Use Cases

* Used to [improve target and biomarker identification and prioritization](https://www.elsevier.com/about/press-releases/elsevier-launches-embiology-to-deliver-the-unparalleled-insights-into-biological-activities-that-accelerate-drug-discovery?) for drug discovery and development, by surfacing relationships between genes/proteins and disease biology via cause-and-effect relationships and filtering by concept types.  
* Used to [visualize and explore relationships](https://www.elsevier.com/about/press-releases/elsevier-launches-embiology-to-deliver-the-unparalleled-insights-into-biological-activities-that-accelerate-drug-discovery?) among drugs, genes/proteins, and disease states via Sankey diagrams and interactive filters to support hypothesis generation for drug repurposing projects.
* Used as the data backbone ([1.4 million entities; \~15.7 million relationships; 87.2 million referenced facts](https://www.elsevier.com/promotions/revealing-biological-connections-and-pathways-with-embiology?)) updated weekly, to enable more up-to-date literature-driven hypothesis generation and evidence inspection for repurposing decisions.

#### Funding / Supporting Programs

* Proprietary product funded, developed and maintained by Elsevier

#### Licensing & Accessibility

* **License**: Proprietary; governed by Elsevier website terms and commercial agreements
* **Accessibility**: Subscription/license required; contact vendor for access and redistribution permissions
* **Note on reuse**: Commercial license required for redistribution

------------------

<a id="primekg"></a>

### PrimeKG (Precision Medicine Knowledge Graph)

- **Homepage**: [https://github.com/mims-harvard/PrimeKG](https://github.com/mims-harvard/PrimeKG)
- **Graph Version/ Release:** See [Matrix release pages](https://docs.dev.everycure.org/releases/release_history/)
- **Current status:** Active, open‑source project with public data repository

#### Rationale for inclusion in EC KG

Disease‑centric, multimodal KG integrating ~20 resources and >4M relationships, harmonized to enable precision‑medicine analyses.
Frequently used as a benchmark and as input to downstream ML for repurposing and disease subtyping.

#### Citations

* **Primary**: `Chandak P. et al. Building a knowledge graph to enable precision medicine. Scientific Data. 2023;10(1):67`, [PMID:36842935]().
* `Perdomo-Quinteiro P. et al. Knowledge graphs for drug repurposing: a review of databases and methods. Briefings in Bioinformatics. 2024 Jul 3;25(4):bbae331`, [PMID:39325460]().

#### Drug Repurposing-related Use Cases

* Used to support drug-disease prediction by including an abundance of ‘indications’, ‘contradictions’, and ‘off-label use’ edges, enabling AI models to explore therapeutic action in less well-covered disease contexts ([PMID:36842935]()).
* Used to improve coverage of rare and common diseases via connections across multiple biological scales (e.g. phenotypes, proteome perturbations, pathway nodes), increasing the graph’s utility for ML/AI models in predicting new repurposing hypotheses ([PMID:36842935]()).  
* Employed in external review articles — for example “Knowledge Graphs for drug repurposing: a review” — as an example of a knowledge graph with strong drug entity coverage and use in inference of drug-gene/disease relations ([PMID:39325460]()).

#### Funding / Supporting Programs

* National Science Foundation ([**IIS-2030459 and IIS-2033384**](https://www.nature.com/articles/s41597-023-01960-3))
* US Air Force Contract ([**FA8702-15-D-0001**](https://www.nature.com/articles/s41597-023-01960-3))
* [Harvard Data Science Initiative](https://www.nature.com/articles/s41597-023-01960-3)
* [Awards from Amazon Research, Bayer Early Excellence in Science, AstraZeneca Research, and Roche Alliance with Distinguished Scientists](https://www.nature.com/articles/s41597-023-01960-3)

#### Licensing & Accessibility

* **License**: [CC0 1.0 for KG](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM); [MIT for code](https://github.com/mims-harvard/PrimeKG?tab=readme-ov-file#license)
* **Accessibility**: Public download via Dataverse (raw KG and largest connected component); programmatic access available
* **Note on reuse**: Dataset licenses and data‑use terms specified on the [Dataverse record](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM) and by individual upstream resources
