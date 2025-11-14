# Note: Equivalence between AUROC and MQR 

In this section, we clarify the definition of mean quantile rank against non-positives $\text{MQR}$ and justify the equation

$$
\text{AUROC} = 1 - \text{MQR}.
$$

In essence, this relationship stems from the fact that the AUROC is equal to the probability that a randomly chosen positive datapoint ranks higher than a randomly chosen negative (Hanley et. al.). 


The *rank against non-positives* for a pair $(d,i)$, denoted by $\text{rank}_{np}(d,i)$,
refers to it's position among non-positive (i.e. unknown or known negative) pairs when sorted by treat score. In other words, it is the rank with any other positive pairs taken out.
The *quantile rank against non-positives* measures the proportion of non-positive pairs that have a lower rank than the pair. It is defined as 

$$
QR_{np}(d,i) = \frac{\text{rank}_{np}(d,i) - 1}{N}
$$

where $N$ is the number of known or known negative pairs. The mean quantile rank against non-positives is given by 

$$
\text{MQR} = \frac{1}{|GT|} \sum_{(d,i) \in GT} QR_{np}(d,i).
$$


To see the relationship between $\text{AUROC}$ and $\text{MQR}$, let $\mathcal{P}$ and $\mathcal{N}$ denote the set of positive and negative pairs respectively. By the aforementioned characterisation of AUROC,

$$
\text{AUROC} = \mathbb{P}_{x \sim \mathcal{P}} \mathbb{P}_{y \sim \mathcal{N}} [\gamma(x) \geq \gamma(y)]
$$

where $\gamma$ denotes the treat score. Then, 

$$
\mathbb{P}_{y \sim \mathcal{N}} [\gamma(x) \geq \gamma(y)]  = \frac{|\{y \in \mathcal{N} : \gamma(x) \geq \gamma(y)\}|}{N} = \frac{N - |\{y \in \mathcal{N} : \gamma(x) < \gamma(y)\}|}{N}
$$

where $N = |\mathcal{N}|$. But $|\{y \in \mathcal{N} : \gamma(x) < \gamma(y)\}|$ is equal to $\text{rank}_{np}(x) - 1$
 so by the above definition of  quantile rank, 

$$
\mathbb{P}_{y \sim \mathcal{N}} [\gamma(x) \geq \gamma(y)] = 1 - \text{QR}(x).
$$ 

Substituting back above shows that the desired equation holds.