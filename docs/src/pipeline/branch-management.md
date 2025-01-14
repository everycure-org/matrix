# Branch Management - Matrix Repository

## Background 
We are developing our MATRIX codebase really quickly and it's evolving quickly, with more than 24 regulary contributors. Whilst we already adhere to good coding standards when it comes to development - we undergo code reviews, have CI/CD in place, document our work - there have been many occassions when bugs 'sneaked into' our codebase, essentially because not all bugs can be detected via CI. Majority of those bugs can be only detected by running MATRIX pipeline e2e on full, real data. However we can't run e2e for each feature/PR as filechanges can be of different nature and scale. Therefore at the moment we accept occassionally introducing bugs which sometimes take a day and other times - weeks, to debug.

## Motivation

We want to reach a point where our main is stable and production ready. This means:
* One can re-branch from main any time and be sure that code there is 100% correct
* During development, one can refer to main as 'ground truth'

This is important because:
* Broken main slows development for everyone.
* We need to be able to execute e2e runs when we have a request, either from medical team or from stakeholders.
* When we open-source, broken CI or faulty product is discouraging for external contributors + bad for reputation.

However, adding additional buffer layer might also increase PR overhead. One solution to prevent it would be to use release tags as stable reference point in the codebase. 

## Release Tags on Main
To ensure stability, we will utilize release tags as reference points in the codebase. Instead of assuming that main is always stable, our approach is to follow release tags. 

This release tag will have been pushed to the **production environment** and will be deployed in production. This means that this tag will reflect our 'status quo' pipeline with best performant model on a validated dataset.

### Production environment

#### Identify The Latest  Stable Release
First, you need to identify the most recent stable release tag (can be found on github matrix page). This is a release tag you can use for branching out for development; essentially you should treat this commit as a production-ready, stable point in the codebase.

All release tags have been checked with real data release pipeline runs and have been confirmed to work without problems.

#### Starting development work
To start development work, you need to branch out from the release tag. You can do it by the following
 ```bash
   git checkout -b feat/foobar v0.2.5
 ```
This will create a `feat/foobar` branch which you can use for development. While you can also branch out from main directly (which is the most natural for most git users), you need to note that branch might be broken/not fully checked.

#### Testing before merging
Once you feature is complete, you should make sure to test it before merging it. While some changes might not require thorough testing (e.g. ruff re-formatting, docs changes, adding comments), pipeline changes should be verified with a sampled real data. Thus before merging your pipeline, you should ensure it runs locally with sampled data.

#### Merging back to main
Once your feature is complete and tested with sampled data, you can merge it back to main. Whilst some bugs might still sneak in (as some bugs will be only detected with full real data), **the main is not production-ready until it has been tagged as a release**.


## Process of creating release (Work In Progress)

Details on how to release can be found in the following [document](../infrastructure/runbooks/01_releases.md)

Before a release is created, one needs to ensure that
1). Data engineering pipeline runs to completion without problems
2). Quality of the output of data engineering pipeline (i.e. KG) is good. 
3). Modelling pipeline runs to completion without problems.
4). Quality of the output of modelling pipeline (i.e. matrix) is good.




