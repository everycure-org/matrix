The matrix transformation pipeline applies changes to the full matrix after it’s generated. At the moment, the only transformation is for frequent flyer normalisation


#### Frequent flyers

Frequent flyers are drugs or diseases that often appear at the top of the matrix because they have unusually high treat scores. This can skew results, making it harder to find less obvious but important pairs.
Ranking-based matrix normalization helps fix this by adjusting scores to reduce the influence of these frequent flyers, giving a fairer ranking to all pairs.

See experiment details [here](https://docs.dev.everycure.org/experiments/2025/matrix_transformation_refined/)


##### Rank based frequent flyer transformation
Let $(d, i)$ denote a drug-disease pair and let:

- $q_{\text{matrix}}(d, i)$ denote the *quantile rank* of the drug-disease pair $(d, i)$ in the matrix.
- $q_{\text{drug}}(d, i)$ denote the *drug-specific quantile rank* of the disease $i$ in the matrix. That is the quantile rank of the disease $i$ in the list of all diseases in the matrix paired with the drug $d$.
- $q_{\text{disease}}(d, i)$ denote the *disease-specific quantile rank* of the drug $d$ in the matrix. That is the rank of the quantile  drug $d$ in the list of all drugs in the matrix paired with the disease $i$.

The normalised matrix is obtained by sorting the matrix pairs by the **rank-normalised score**, which is defined as

$$
t(d, i) = w_{\text{matrix}} q_{\text{matrix}}(d, i)^{-\alpha} + w_{\text{drug}} q_{\text{drug}}(d, i)^{-\beta} + w_{\text{disease}} q_{\text{disease}}(d, i)^{-\gamma  }
$$

where

-  $w_{\text{matrix}}$, $w_{\text{drug}}$, $w_{\text{disease}} \geq 0$ are the *weight parameters* assigned to the matrix-wide, drug-specific and disease-specific ranks respectively, and
- $\alpha, \beta, \gamma >0$ are the *decay parameters* controlling the emphasis on the top ranks for matrix-wide, drug-specific and disease-specific rankings respectively (the larger they are, the more emphasis on the top ranks).


##### Almost pure rank based implementation

An effective simplification of the rank-normalised score is referred to as *almost pure hyperparameters* and is described by the following formula: 

$$
t(d, i) = \varepsilon q_{\text{matrix}}(d, i)^{-\gamma} +  q_{\text{drug}}(d, i)^{-\gamma  } +  q_{\text{disease}}(d, i)^{-\gamma  }
$$

where: 

-  we have a single decay parameters $\gamma > 0$,
-  the matrix weight parameter $\varepsilon > 0$ is intended to be set to a small value. The purpose of the matrix weight is to break ties between pairs which have the same drug and disease specific rankings. $\varepsilon is set to an arbitrarily small value.

> These are set to the default $\varepsilon =  0.01$ and $\gamma = 0.05$.


!!! info
    Note that the hyperparameters were determined experimentally outside the pipeline and are statically set in the pipeline parameters. We do not plan to perform hyperparameter optimization within the pipeline at this stage, as it is complicated to create a separate validation set (distinct from the test set) after matrix generation. In the future, we may further adjust these parameters based on medical preferences for the score distribution.
