## Status

In Review

## Context

At the time of writing, the MATRIX pipeline runs are all scheduled on the standard nodes. As the MATRIX Pipelines are fault-tolerant workloads, we do not need it to be scheduled on the standard nodes.

As part of our GreenOps Initiative, we would like to run our fault-tolerant, not critical workloads on GKE Spot Instances.

## Problems

• Inefficient Resource Utilization – All MATRIX pipeline runs are currently scheduled on standard GKE nodes, which are provisioned for high availability and reliability. This results in underutilization of cheaper, ephemeral resources suitable for non-critical workloads.
• Missed Sustainability Goals – Continuing to use standard nodes for non-critical workloads increases the cluster’s energy footprint, making it harder to align with the GreenOps Initiative objectives of reducing carbon and energy impact.
• Lack of Scheduling Optimization – No current mechanism exists to prioritize Spot Instances for MATRIX workloads, meaning there is no guarantee that these workloads will benefit from lower-cost, lower-priority compute.

## Approach

1. We'll spin up Spot Instances Node Pools in our GKE cluster. 
2. We will modify our pods to be placed
    - On the spot instance.
    - If there is no spot instance available, then we will schedule it normally on the standard instance.
3. The pod will begin running on the spot instance.
4. Incase of termination, the pod will automatically re-run on the standard instance

### Pros

•	Improved Sustainability – Supports the GreenOps Initiative by utilizing resources more efficiently and reducing the carbon footprint.
•	Scheduling Flexibility – Ensures workloads first attempt to run on Spot Instances while retaining the ability to fall back to standard nodes if required.
•	Fault-Tolerance Alignment – Matches workload characteristics (fault-tolerant, non-critical) with the most cost-effective compute option without compromising reliability.
•	Operational Continuity – Automatic re-scheduling on standard nodes in case of Spot Instance termination ensures no manual intervention is required.

### Cons

•	Instance Preemption Risk – Spot Instances can be terminated at any time by GCP, potentially interrupting job execution, terminating instances in 30 seconds and increasing total run time.
•	Potential for Longer Completion Times – Frequent Spot Instance interruptions may cause workloads to re-run more often, delaying overall pipeline completion.
•	Increased Scheduling Complexity – Requires additional configuration (tolerations, node affinity, fallback logic) to ensure correct workload placement.
•	Monitoring Overhead – Spot Instance usage may require enhanced monitoring to detect and handle preemptions effectively.
•	Variable Cost Savings – Cost reductions depend on Spot Instance availability; savings may fluctuate if standard node fallback is frequently used.

## Decision

We Proceed with implementing Spot Instance Node Pools in the GKE cluster for MATRIX pipeline workloads.
We will configure pod scheduling to prefer Spot Instances while retaining the ability to fall back to standard nodes if Spot capacity is unavailable or preempted.
This approach balances cost savings, sustainability goals, and operational continuity, while accepting the manageable risks of preemption and slightly longer completion times for non-critical, fault-tolerant workloads.