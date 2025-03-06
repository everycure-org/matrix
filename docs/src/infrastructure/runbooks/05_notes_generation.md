---
title: Release Article Generation by AI
---

## Overview

This runbook outlines the cases in which the release article is generated from the GitHub Actions workflow of kedro pipeline submission. The golden rule is that the article is generated only when there is a **major** or **minor** bump.

## Examples
Given that the latest minor release is v0.2.5:

1. **Major bump**  
    - New release: v1.0.0 (major bump). Article generation is expected.
    
2. **Minor bump**  
    - New release: v0.3.0 (minor bump). Article generation is expected.
    
3. **Patch bump**  
    - New release: v0.2.8 (patch bump). Article generation is skipped.
    
4. **Patch downgrade**  
    - New release: v0.2.3 (patch downgrade). Article generation is skipped.  
    - Heads up: If the release version order is v0.2.5 → v0.3.2 → v0.3.0, **v0.3.2** is considered a minor bump, as it is triggered first, and **v0.3.0** is considered a patch downgrade, as it follows v0.3.2.
    
5. **Major or minor downgrade**  
    - New release: v0.1.8 (minor downgrade). This is forbidden and prevented from `kedro submit`. The same applies to major downgrades.
The reason to forbid it is that is a bit weird: yes, the Python programming language allows it, but for good reasons. Whereas we might add bugfixes to older releases, all new feature development should continue from the latest official release. processing the commits and PRs that have lead to this intermediate release, while not taking into account features that were added "in the future" (relative to this intermediate release) is harder if a major or minor release would be created that isn't the latest, then the release notes and article of the already existing release would become unlogical, since they are written relative to the last official (major or minor) release. An intermediate (major or minor) release would need to mention contributions that were already included in the latest release, and would need to remove it from that one too.

## Best Practices

- Ensure all PRs are correctly labeled and titled before generating release notes.
- Review and update the release notes draft for clarity and completeness.
- Acknowledge all contributors to foster a collaborative environment.