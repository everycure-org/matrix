# Neo4J Debugging

## Machine out-of-memory

We've encountered a situaton where the Neo4J instance went out-of-memory. The data is of Neo4J is housed inside the PVC on the cluster. This manual describes how to [clear out the log files](https://neo4j.com/developer/kb/how-do-recover-from-no-space-left-on-device/).

First, set the replicas of the Neo4J statefulset to zero:

```
kubectl -n neo4j scale statefulset neo4j --replicas=0
```

Secondly, create a test-pod and attach the PVC. Create a file called `pod.yml` with the following contents:

```yaml
apiVersion: v1
kind: Pod
metadata:
  namespace: neo4j
  name: temp-pod
spec:
  containers:
  - name: temp-container
    image: busybox
    command: [ "sleep", "3600" ]
    volumeMounts:
    - mountPath: /mnt/data
      name: temp-storage
  volumes:
  - name: temp-storagel
    persistentVolumeClaim:
      claimName: data-neo4j-0
```

Next, apply the pod:

```bash
kubectl apply -f pod.yml
```

Finally, exec into the pod and clear out the log files:

```bash
kubectl -n neo4j exec -it temp-pod -- /bin/sh
```


