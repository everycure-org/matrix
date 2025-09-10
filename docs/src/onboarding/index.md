---
title: Onboarding
---




Welcome to the Matrix onboarding guide! This guide provides an introduction to the project and the codebase, and guides you through the process of setting up your local environment.

!!! success
    As you are walking through the onboarding guide, please create an [onboarding issue](https://github.com/everycure-org/matrix/issues/new?assignees=&labels=onboarding&projects=&template=onboarding.md&title=%3Cfirstname%3E+%3Clastname%3E) in the codebase for visibility by the codeowners. Then continue through this onboarding journey.
    

## Why Every Cure?

<iframe width="640" height="390" src="https://www.youtube.com/embed/3ElaCVvDZfI?si=lk3b1rSMutyiierm" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

## Why Matrix?

<iframe width="640" height="390" src="https://www.youtube.com/embed/67_Z40Ap1pU?si=XlCu7fBHxxkBTchH" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

<details> <summary> Podcast summary before you dive in </summary>

  In this episode of "Infinite Loops," Dr. Grant Mitchell, a medical doctor with numerous qualifications, discusses his company Every Cure and its mission to utilize AI for drug repurposing. He explains that many existing drugs, particularly off-patent ones, have the potential to treat a vast number of diseases, but traditional pharmaceutical companies lack the incentive to explore these possibilities. Mitchell shares the motivating story of how he and his co-founder, David Fagenbaum, who survived a rare disease through drug repurposing, founded Every Cure to systematically unlock the potential of existing drugs using artificial intelligence.

  The conversation delves into the innovations and strategies employed by Every Cure, including constructing a comprehensive biomedical knowledge graph and using AI to identify repurposing opportunities. Mitchell emphasizes the importance of combining AI with clinical expertise to efficiently repurpose drugs and help patients quickly. He also highlights Every Cure's nonprofit model, which facilitates collaboration and open data sharing to maximize the impact of their research. Through significant philanthropic support and a strategic agreement with ARPA-H, Every Cure aims to save millions of lives by bringing effective treatments to neglected diseases, demonstrating a powerful intersection of technology, medicine, and altruism.
</details>

## Role and responsibilities

### Every Cure MATRIX SteerCo

- Provide strategic alignment to Working Groups to ensure ARPA-H obligations are met
- Maintain MATRIX Project Management Infrastructure
- Onboard new MATRIX members
- Propose and negotiate necessary contractual modifications to ARPA-H Team
- Resolve / troubleshoot roadblocks raised by Scientific Leads
- **MATRIX: SteerCo**
    - Project Manager = Charlie Hempstead, Every Cure
    - Technical Lead = Matej Macak, Every Cure Technical Advisor
    - MATRIX Principal Investigator = David Fajgenbaum, Every Cure

### Scientific Leads - Working Groups

- Chair weekly Working Group meetings
- Propose modifications / additions to Workstreams to achieve Working Group deliverable
- Resolve / troubleshoot roadblocks raised by Workstream Leads
- Escalate barriers / roadblocks to Working Group deliverable completion in a timely manner to MATRIX Working Team
  
**Working Group 1: Data and Knowledge Graph**

  - Project Manager = Carrie Pasfield

  - Scientific Lead = Chris Bizon

**Working Group 2: Modeling and Evaluation**

  - Project Manager = Robyn Macrae

  - Scientific Lead = Andrew Su

### Workstream Leads

- Responsible for Workstream deliverable(s)
- Resolve / troubleshoot roadblocks raised by Workstream Contributors
- Escalate barriers / roadblocks to Workstream deliverable completion in a timely manner to Scientific Lead

### Workstream Contributors

- *N.B. A Workstream Contributor can also be a Workstream Lead*
- Responsible for Workstream sub-deliverable(s)
- Execute of day-to-day tasks
- Escalate barriers / roadblocks to sub-deliverable completion in a timely manner to Workstream Lead**

## Communication 

### Slack for communications

We use slack channels for day 2 day communication. These channels are prepared to all:

- **matrix-wg-data-kg:** Data and KG working group
- **matrix-wg-modelling:** Modelling and Evaluation working group
- **matrix-all:** Matrix wide chat
- **matrix-interesting:** Matrix wide interesting information to share. Used for papers, external sources etc that are worth sharing
- **matrix-updates:** Updates chat for sharing regular structured updates from each WG

You can create your own of course, although it would be great if we can keep the communication transparent 

In terms of our communication principles, we utilize threads within our discussions to keep conversations organized and to ensure that everyone involved can easily follow along. If we need to specifically address someone, we tag them directly. This method helps in maintaining a clear line of communication and ensures everyone's thoughts are heard and accounted for. However, we are also mindful of avoiding any unnecessary noise within the channel. To this end, we avoid using "at channel" or similar blanket notifications unless absolutely necessary, so as not to disturb everyone with irrelevant information. This way, we can maintain a productive and respectful communication environment.

### Documentation and Information Sharing

- Documentation is in markdown, we use [mkdocs material](https://squidfunk.github.io/mkdocs-material/) for structuring our documentation + use Drawio for drawings using `drawio.svg` extension
- For other file types, use Google Drive. E.g. for Powerpoint or Excel files. You can also use Google Collabs to share collaborative notebooks with each other if helpful.

### Calls & Digital Whiteboarding

We use Slack and Zoom for our communication. Either huddles or Zoom calls. Ad hoc whiteboarding can be easily done with [excalidraw.com](http://excalidraw.com).

[Proceed to the installation guide :material-skip-next:](./installation.md){ .md-button .md-button--primary }
