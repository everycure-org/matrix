# Data versioning workflow

It's a tale of 2 orchestrators: Argo Workflow, to monitor tasks running on k8s,
and GitHub Actions, to act on the repository where this code is hosted.

## Current implementation

```mermaid
sequenceDiagram
	actor eng as Engineer
	box rgb(40,100,50, .5) ns:argo-workflows
	participant ws as Workflows Server
	end
    eng->>ws: kedro submit --pipeline data_release
    create participant dr as Data Release
  box rgb(40,50,100, .5) ns:data-release
		participant es as Event Source
		participant eb as Event Bus
		participant esen as Event Sensor
	end

    ws->>dr: start workflow
    dr->>+dr: launch pods running kedro nodes
    destroy dr
		es-xdr: 👀 Detect done (filter)!
    es->>eb: FWD 📨!
    esen->>eb: “filter”
	  note right of esen: trigger an action
```

```mermaid
sequenceDiagram
	participant es as Event Sensor


	create participant curl as Pod running "Curl"
	es ->> curl: Come alive
	actor emp as Employee
	actor emp2 as Employee2
	participant gh as GitHub
	curl ->> gh: HTTP POST to dispatch<br />{"release": "v0.1.2", "fingerprint": "00fe900d"}
	create participant gha as GitHub Actions
	gh->>gha: delegate
	gha->>gha: git checkout <fingerprint>
	gha->>gha: git tag <release>
	gha->>gha: setup matrix cli
	gha->>gha: gcloud auth
	gha->>gha: generate release notes & article
	gha->>gha: add, commit & push notes, article & tag
	gha->>gh: create PR (label: Release)

	gh ->> emp: request review
	emp ->> gh: comments & modifies
	emp2 ->> gh: approves PR
	create participant gha2 as GitHub Actions'
	gh ->> gha2: notifies PR done
	gha2 ->> gha2: git checkout
	gha2 ->> gh: create release using AI post
```

The above combination makes the Git lineage end result look like:

```mermaid
    gitGraph
       commit
       commit
       commit id: "kedro submit" tag: "v1.0.0"
       commit id: "add release notes and article"

```

## Future work

A data quality check can be implemented as a required pipeline after
data-release or as an optional one in kedro (branching off), but required for
the release process (e.g. sensor→DQ→sensor→curl).

## References

https://docs.dev.everycure.org/infrastructure/argo_workflows_locally/
https://kind.sigs.k8s.io/docs/user/quick-start  
https://argoproj.github.io/argo-events/installation/  
