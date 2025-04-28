# Biomedical Knowledge Graphs
Our MATRIX pipeline is built to accept a several different Knowledge Graphs (KGs) both independently and in an integrated fashion. The KGs currently supported e2e by the pipeline are specifid in [`settings.py`](../../../../pipelines/matrix/src/matrix/settings.py) where one can specify boolean flags for whether a specific dataset should be integrated into the final EC KG or not.

The Knowledge Graphs ingested in our pipeline are compliant with KGX format.

## Public KGs

### RTX-KG2
RTX-KG2 is a Biomedical Knowledge Graph created by [] in. For more information, see: 

Versions within our system: 
  * v2.7.3
  * v2.10
  * v2.10-validated

### Robokop
ROBOKOP is a Biomedical Knowledge Graph created by []. For more information, see: 
Versions within our system: 
  * 30fd1bfc18cd5ccb

## Proprietary KGs
Our system is also ingesting proprietary datasets which are stored in a separate, secure environment, ensuring the access is granted only to the authorized users.

### SPOKE
Spoke is a Biomedical Knowledge Graph created by []. For more information, see: 

Versions within our system: 
  * v5.2 

### EmBiology
EmBiology is a knowledge Graph created by Elsevir. For more information, see:

!!! info "EmBiology KG Preprocessing"

  Note that raw EmBiology Knowledge Graph is **not compliant with KGX format**, which is the default format followed in our pipeline. Therefore, EmBiology KG that is being ingested in our MATRIX pipeline has been transformed & preprocessed to fit our system - for exact transformations, see `preprocessing` pipeline tagged with `embiology-kg`. The versions of Embiology KG specfied in our system refer to the version preprocessed by our system.

Versions within our system: 
  * v0.1 - first preprocessed & KGX compliant version 

## Data Products Maintainer Team
Data Products Team is working towards curating high-quality 


| Contributor | Organisation | Position | 
| ----------- | ----- | ----- |
| En May Lim | Every Cure; Data Products Team | Product Owner|
| Jane Li | Every Cure; Data Products Team | Tech Translator |
| Jacques Vergine|  Every Cure; Data Products Team | Sr Machine Learning Engineer |
| Piotr Kaniewski | Every Cure; Data Products Team | Data Scientist |
| Nico M |  Monarch Initiative; Data Products Team | Data Scientist |

## Knowledge Graph Maintainer Team
_Note that contacts below are team members who contributed by providing mentioned knowledge graph in an accepted KGX format; they are not necessarily the creators or curators of these graphs however they might be able to direct you to the right person if needed_

| Contributor | Organisation | KG |
| ----------- | ------------ |------------ |
| Chunyu Ma | RTX-KG2 | ------------ |
| Kathleen Carter | ROBOKOP | ------------ |
| Charlotte Nelson  | SPOKE | ------------ |
| Jacques Vergine & Piotr Kaniewski| EmBiology | ------------ | 