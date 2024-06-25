## Model Evaluation for Drug Repurposing Models

This article will review the various evaluation methods used to assess the performance of drug
repurposing prediction models.

### Introduction

Drug repurposing prediction models are machine learning (ML) models designed to identify potential
new uses for existing drugs. These models take a drug and a disease as input and output a "treat"
score, a number between 0 and 1 representing the likelihood of the drug treating the disease.

### Evaluating Model Performance

Before deploying a drug repurposing model, it's crucial to evaluate its performance. This involves
determining how well the model performs on tasks like:

- Identifying a drug for a specific disease
- Discovering repurposing opportunities across all drugs and diseases

Since there are multiple ways to utilize these models, we employ several evaluation metrics to
capture different aspects of model performance.

### Types of Evaluation Metrics

We currently use five main types of evaluation metrics:

1. **Disease-Specific Ranking Metrics**
2. **Ranking Over All Drugs and Diseases**
3. **Classification Metrics**
4. **Stability Metrics**
5. **Medical Metrics**

#### 1. Disease-Specific Ranking Metrics

These metrics assess how well the model ranks drugs for a specific disease. The goal is to determine
if the model can accurately identify drugs that are known to treat a given disease by ranking them
high on the list.

Two common disease-specific ranking metrics are:

- **Hit@k:** Measures the probability that a known positive drug (a treatment for the disease) ranks
  among the top k drugs in the list, where k is a chosen integer. For example, if Hit@10 is 0.6, it
  means there's a 60% chance that a given positive drug is within the top 10 ranked drugs for that
  disease.
- **MRR (Mean Reciprocal Rank):** Calculates the expected value of 1/rank for a positive drug. For
  instance, an MRR of 0.1 suggests the model is expected to rank a positive drug roughly 10th on the
  list (since 1/10 = 0.1).

#### 2. Ranking Over All Drugs and Diseases (AUROC)

When we want to evaluate the model's ability to identify repurposing opportunities across all drugs
and diseases, we use the **AUROC (Area Under Receiver Operator Curve)** metric.

Imagine a matrix where rows represent drugs and columns represent diseases. Cells are marked as
positive if the drug treats the disease and negative otherwise. AUROC approximates the probability
that a randomly chosen positive drug-disease pair ranks higher than a randomly chosen negative
drug-disease pair.

For example, an AUROC of 0.9 implies that a positive drug-disease pair typically ranks higher than
90% of negative pairs.

**Note:** Computing AUROC involves certain complexities:

- We don't know all positive and negative drug-disease relationships.
- Therefore, we use **synthetic negatives**, leveraging the fact that a randomly chosen drug is
  highly unlikely to treat a randomly chosen disease.

#### 3. Classification Metrics

Classification metrics complement ranking metrics by focusing on the model's ability to classify
drug-disease pairs as "treat" or "not treat" based on a specific threshold. This is particularly
important for evaluating performance on tricky drug-disease pairs where the true relationship might
be ambiguous.

We primarily use two classification metrics:

- **Accuracy:** Represents the proportion of the dataset that the model correctly classifies as
  "treat" or "not treat". For example, an accuracy of 0.7 means 70% of the drug-disease pairs in the
  dataset are correctly classified.
- **F1 score:** Similar to accuracy, but it takes into account class imbalances. This is crucial
  because our dataset typically has more known negatives than known positives. For instance, if F1
  is 0.7, it signifies that 70% of drug-disease pairs the model flags as "treat" are indeed known
  positives, and 70% of known positives are identified by the model.

#### 4. Stability Metrics

Stability metrics assess the robustness of the model's outputs to slight changes in model
architecture, data, or the underlying knowledge graph (KG). This is important because unstable
outputs suggest a noisy model, potentially leading to unreliable predictions.

We measure stability using **commonality metrics**. These metrics focus on the changes in the
top-ranked drug-disease pairs while ignoring possible noise in the probability scores of
lower-ranked drugs.

**Commonality** measures the average proportion of drugs that remain in the top k for a randomly
chosen disease, where k is a randomly chosen integer less than 100. For example, a commonality of
0.2 suggests that we expect approximately 20% of the top 5/10/100 drugs to remain in the top ranking
after introducing slight changes to the model, data, or KG.

**Weighted commonality** is similar to commonality but places more emphasis on stability near the
top of the ranked list.

#### 5. Medical Metrics

The metrics discussed so far are robust in a statistical sense because they are computed with large
amounts of data. However, they can only approximate the true performance of the model in real-world
medical applications.

To address this limitation, we supplement our metrics with specific medical examples and seek domain
expert opinions from our KOL (Key Opinion Leader) team. This involves:

- **Sanity checks:** Evaluating the model's performance on recently repurposed drugs to assess its
  ability to identify emerging treatment relationships.
- **Specific medical examples:** Examining how the model ranks drugs for specific diseases,
  particularly those with recently discovered repurposing applications. Examples include:
- Colchicine for heart attacks
- Minocycline for FOP
- Xolair for food allergies

### Future Evaluation Metrics

We are actively developing new evaluation methods to further enhance our assessment of model
performance. Our current focus includes:

- **Time split evaluation:** This method assesses model performance on drug-disease relationships
  discovered after the publication date of the knowledge graph and training dataset used for the
  model. This will involve analyzing recent clinical trial data and incorporating more
  "sanity-check" type examples.
- **Off-label drug treatment data:** We plan to evaluate model performance on repurposed drug
  treatments already in use but not officially approved. This data is available in medical records
  and literature, providing valuable insights into real-world drug repurposing applications.
- **Simulate rare diseases:** Recognizing the limited data associated with rare, uncured diseases,
  we are exploring methods to simulate these conditions by removing KG edges connecting to test
  diseases with known treatments. This approach will help us assess the model's ability to identify
  repurposing opportunities for challenging diseases with limited data.

### Commentary

The presenter provided a comprehensive overview of model evaluation methods for drug repurposing.
However, certain aspects could be clarified or expanded upon:

- **Specificity of MRR:** While the presenter mentioned that MRR measures the expected value of
  1/rank for a positive drug, it's important to specify that this calculation is done for a specific
  disease. So, an MRR of 0.1 for a given disease means the model is expected to rank a positive drug
  for that disease roughly 10th on the list.
- **Details of AUROC computation:** The presenter briefly mentioned the use of synthetic negatives
  in AUROC computation. It would be beneficial to elaborate on how these negatives are generated and
  incorporated into the calculation.
- **Concrete examples for stability metrics:** While the presenter explained the concept of
  commonality, providing specific examples of how commonality is affected by slight changes in model
  architecture, data, or KG would enhance understanding.
- **Quantitative medical metrics:** The presenter primarily focused on qualitative medical metrics
  based on expert opinions and specific examples. Exploring quantitative medical metrics, such as
  the number of correctly predicted repurposing opportunities for a set of recently discovered
  drug-disease relationships, could provide a more objective assessment of model performance in a
  medical context.

By incorporating these suggestions, we can further improve the clarity and comprehensiveness of our
documentation on model evaluation for drug repurposing models.
