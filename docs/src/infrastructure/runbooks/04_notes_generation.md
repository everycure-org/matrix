---
title: Notes and Article Generation by AI
---

## Overview

This runbook outlines the cases in which notes and articles are generated. The golden rule is that notes and articles are generated only when there is a **major** or **minor** bump.

## Examples
Given that the latest minor release is v0.2.5:

1. **Major bump**  
    - New release: v1.0.0 (major bump). Notes and article generation are expected.
    
2. **Minor bump**  
    - New release: v0.3.0 (minor bump). Notes and article generation are expected.
    
3. **Patch bump**  
    - New release: v0.2.8 (patch bump). Notes and article generation are skipped.
    
4. **Patch downgrade**  
    - New release: v0.2.3 (patch downgrade). Notes and article generation are skipped.  
    - Heads up: If the release version order is v0.2.5 → v0.3.2 → v0.3.0, **v0.3.2** is considered a minor bump, and **v0.3.0** is considered a patch downgrade.
    
5. **Major or minor downgrade**  
    - New release: v0.1.8 (minor downgrade). This is forbidden and prevented from `kedro submit`. The same applies to major downgrades.

## Best Practices

- Ensure all PRs are correctly labeled and titled before generating release notes.
- Review and update the release notes draft for clarity and completeness.
- Acknowledge all contributors to foster a collaborative environment.