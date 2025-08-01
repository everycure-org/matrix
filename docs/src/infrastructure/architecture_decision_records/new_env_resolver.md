---
title: Platform Refactor and Standardization.
---

## Status

Draft

# Context

As the MATRIX project has grown in the number of datasets it consumes from different sources, and the with the goal to open-source it, the initially thought design for the variable resolution of the environment that determines the pipeline run and datasets sources, has become too complex. As we have introduced a public bucket alongside `dev` and `prod` bucket, switching between the three has proved that there is tight coupling between environments (base, cloud, and test). We aim to reduce the cognitive load and manual intervention that is needed to avoid human error.

In the past, we had different needs when creating datasets and storing them in the GCP bucket and hence it has led to having different folder structure (`data/kedro/01_raw/<datasets>`, `data/01_raw/<datasets>`, `data/01_raw/KG/<datasets>`). Having different folder structure has made it harder to have a unified datasets resolver and causes a lot of confusion. We would now like to standardize it.

## Current Environment Resolution Structure

```mermaid
graph TD
    A[Application Start] --> B["load_environment_variables()"]
    B --> C[Load .env.defaults]
    C --> D[Load .env - overwrites defaults]
    D --> E[Kedro OmegaConfigLoader]
    
    E --> F[Load base/globals.yml]
    E --> G[Load environment-specific globals]
    E --> H[Load catalog files from subdirectories]
    
    F --> I[Base Configuration]
    I --> J[hardcoded defaults]
    I --> K["${oc.env:VAR,default} resolvers"]
    
    G --> L[Environment Overrides]
    L --> M[test/globals.yml]
    L --> N[cloud/globals.yml] 
    L --> O[sample/globals.yml]
    
    H --> P[Catalog Files - NO Base Inheritance]
    P --> Q["${globals:paths.*} references"]
    P --> R1[Each catalog.yml defines own YAML anchors]
    P --> R2[3 individual catalog files with local definitions]
    
    K --> R[Runtime Values]
    N --> R
    M --> R
    O --> R
    Q --> R
    R1 --> R
    R2 --> R
    
    R --> S[Final Configuration]
    
    subgraph "Environment Variables Sources"
        T[.env.defaults<br/>73 lines of defaults]
        U[.env<br/>User overrides]
        V[OS Environment]
    end
    
    subgraph "Configuration Files"
        W[base/globals.yml<br/>91 lines]
        X[cloud/globals.yml<br/>~50 lines]
        Y[test/globals.yml<br/>33 lines]
        Z1[NO base/catalog.yml exists]
        Z2[3 individual catalog.yml files<br/>Each in own subdirectory]
    end
    
    subgraph "Actual Catalog Pattern"
        CA[base/integration/catalog.yml<br/>Defines _spark_parquet, _pandas_parquet]
        CB[cloud/integration/catalog.yml<br/>Defines _bigquery_ds, overrides datasets]
        CC1[test/filtering/catalog.yml<br/>Defines _spark_json locally]
        CD[Each catalog defines own anchors independently]
    end
    
    subgraph "Resolvers"
        AA[oc.env resolver]
        BB[get_kg_raw_path_for_source]
        CC[merge_dicts]
        DD[if_null]
        EE[cast_to_int]
    end
    
    T --> B
    U --> B
    V --> AA
    W --> F
    X --> G
    Y --> G
    Z2 --> H
    AA --> K
    BB --> K
    CC --> K
    DD --> K
    EE --> K
    
    CA --> R2
    CB --> R2
    CC1 --> R2
    CD --> R2
    
    style A fill:#ff9999
    style S fill:#99ff99
    style T fill:#ffcc99
    style U fill:#ffcc99
    style V fill:#ffcc99
    style Z1 fill:#ffaaaa
    style CA fill:#ffffcc
    style CB fill:#e6ffe6
    style CC1 fill:#e6ffe6
    style CD fill:#ffcccc
    classDef largeFontSize font-size:16px;
    class A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z1,Z2,CA,CB,CC1,CD,AA,BB,CC,DD,EE largeFontSize;
```

### Current Issues Illustrated

1. **Multiple Override Layers**: Configuration flows through 4+ layers (.env.defaults → .env → base/globals → env-specific globals)
2. **Redundant Definitions**: Same values defined in multiple places (e.g., GCS buckets in .env.defaults, base/globals.yml, and cloud/globals.yml)
3. **Complex Resolution**: Custom resolvers handle environment variable fallbacks with hardcoded defaults
4. **Tight Coupling**: Environment-specific globals duplicate base configurations
5. **Inconsistent Path Structures**: Multiple folder patterns for raw data storage
6. **No Catalog Inheritance**: Each of the 3 catalog.yml files defines its own YAML anchors (_spark_parquet, _bigquery_ds, etc.) independently, leading to massive duplication.
7. **Environment-Specific Catalog Overrides**: Cloud and test environments completely redefine catalog datasets rather than inheriting from base.

# Decision

## Environment Variable Resolution

We will create a user script that would contain all of the env needed to run the project and would be our single source of truth. Only the versions of the datasets would be in the `global.yml` file. It would allow the user to easily switch between different environment without having the need to manually enter the environment variable into the `.env` file or rely on the `.env.defaults`.

## Datasets Standardization

We will move all of our datasets into a single folder structure `data/raw` for raw data and update the variables to point to the correct folders.

## Proposed Future State

```mermaid
graph TD
    subgraph "Setup"
        A[User Script<br/>Sets environment context & runtime variables]
        A1[Variables file generated<br/>Contains all runtime config]
    end

    subgraph "Kedro Configuration Loading"
        B[Kedro OmegaConfigLoader]
        C[Custom env_file resolver<br/>Reads from variables file]
        D[base/globals.yml<br/>Static project defaults + dataset versions]
        E[base/catalog.yml<br/>Shared YAML anchors & dataset types]
        F[Environment catalog overrides<br/>cloud/*, test/* catalogs inherit from base]
    end

    subgraph "Final Merged Configuration"
        G[Final Configuration Object<br/>Clean, standardized paths]
    end

    A --> A1
    A1 --> C
    B --> C
    B --> D
    B --> E
    B --> F
    D --> G
    E --> F
    F --> G
    C --> G

    subgraph "Key Changes"
        H[Eliminate .env.defaults]
        I[Eliminate environment globals duplication]
        J[Standardize all paths to data/raw/*]
        K[Create base/catalog.yml with shared anchors]
    end

    style A fill:#99ccff
    style A1 fill:#99ccff
    style D fill:#ffffcc
    style E fill:#ccffcc
    style F fill:#e6ffe6
    style G fill:#99ff99
    style H fill:#ffaaaa
    style I fill:#ffaaaa
    style J fill:#ccffcc
    style K fill:#ccffcc
```

### Benefits of Proposed Structure

1. **Single Source of Truth**: All configuration defaults in `base/globals.yml`
2. **User Script**: Simple environment selection without manual `.env` editing
3. **Minimal Environment Variables**: Only secrets and deployment-specific values
4. **Standardized Paths**: Consistent `data/raw/*` structure across all datasets
5. **Reduced Complexity**: Fewer override layers and resolution paths
6. **Catalog Inheritance**: edit the `ingestion/base/catalog.yml` with proper shared YAML anchors that the 2 catalog files can inherit from, eliminating massive duplication.
7. **Separation of concerns**: Globals file would be limited to just having the versions number and fully owned by the team responsible.

## Implementation Changes Required

1. Create User Environment Script.
2. Remove Variables from `globals.yml`.
3. Create the variable_resolver.
4. Modify base/catalog.yml with Shared Anchors and use the variable_resolver
5. Write proper unit tests.
6. Make sure that the test catalog doesn't write to dev or prod buckets.
7. Update Child Catalogs to Use Inheritance.
8. Standardize All Paths by moving the files to the decided folder structure.
9. Modify the variable file to point to the new location.

## Nice to have feature

1) Adding checksum to verify that the config file has not changed depending on the environment. This would be before running the workflow and also on-cloud.
2) Refactor the `settings.py` file to remove the dynamic mapping to the variable file as well.

## Consequences

This approach would eliminate:
- `.env.defaults` file (73 lines)
- Environment variable duplication across globals files
- Manual `.env` file editing
- individual YAML anchor definitions in catalog files and keep the ones that is absolutely needed.

This would remove the cognitive load on the developer and greatly reduce human errors.