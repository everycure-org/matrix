# Memory debugging for Embeddings node

## Provisioning correct machine types

Initially, we noticed the machine type was not being allocated. This due to the emphemeral storage requirements not being satisfied by any of the nodes.

> ✅ Adding local SSDs for emphemeral storage is supported through the [GCP Terraform module](https://registry.terraform.io/modules/terraform-google-modules/kubernetes-engine/google/latest/submodules/private-cluster) using the `local_ssd_ephemeral_storage_count` attribute, but unfortunately is not available for all machine types. It turns out that the [N2D machine type](https://cloud.google.com/compute/docs/general-purpose-machines#n2d_machines) is one of the types that supports local SSDs. 

## Bumping memory spec

Next, we noticed the Neo4J container going OOM. It turned out this was due to the hardcoded values in the Neo4J container, essentially capping it's resources.

> ✅ This was solved by using the memory requirements as specifed in the pod's limits when setting up the environment variables for the Neo4J container.

![](./assets/memory_usage.png)

```yaml
# Neo4J container
env:
    ...
    - name: NEO4J_dbms_memory_heap_initial__size
      value: "100G"
    - name: NEO4J_dbms_memory_heap_max__size
      value: "120G"
```

As the next step, we set the above NEO4J settings same as the pod's memory request. This resulted in the main container stuck at "Waiting for neo4j to be ready..." with the Pod's status "OOM Killed".
To fix this, we applied 70% of the pod's memory request as the neo4j heap size settings, which solved this problem.

`"{{= sprig.int(inputs.parameters.memory_limit) * 0.7 }}G"`

Moreover, we noticed a similar problem in the Spark configuration, where the the Spark configuration was hardcoded to a specific value.

> ⛔️ This still requires a fix, in the ideal case we would expect the resouces defined in the node to correctly propagate.

```yaml
# spark.yaml
...
spark.driver.memory: 30g
```

## Setting memory spec for Kedro container

Next, we've bumped the requests for Spark in the main container to 50Gb. This resulted in an OOM memory, presumably because Spark had been configured to use all the pods' memory, and when the Kedro/Python process started to use RAM, the container was killed.

```yaml
spark.driver.memory: 50g
```

It's therefore important to leave a buffer between the memory configured for Spark, and the total memory allocated to the container.

```yaml
    # Argo workflow template
    ...
    name: neo4j
    podSpecPatch: |
      containers:
        - name: main
          resources:
            requests:
              memory: 50Gi
            limits:
              memory: 50Gi
```

## Follow-ups

1. Ensure Spark Memory can be configured using environment variable
2. Deep dive into Neo4J Spark connector to learn why so much memory is used
3. (After few runs) investigate memory profile and right size the nodes