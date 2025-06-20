The modelling pipeline trains prediction models using drug-disease pairs and knowledge graph embeddings. 

Key steps implemented include: 

1. *Prepare ground truth dataset*. Load ground truth positive and negative drug-disease pairs. Perform test-train split using a stratified approach which includes the ability to stratify by drug. This enhances the training-test split, ensuring that the test set is representative of the training set. We also implement k-fold cross-validation where by several different train-test splits, referred to as "folds", are created. We also create a "full" split, where the entirety of the ground truth data is used for training the final model.


2. *Synthesise additional training data*. Synthesise additional drug-disease pairs for training using an appropriate sampling strategy.  This is an important step as it allows for the training of models that are not only representative of the training data but also generalise to new, unseen data.

3. *Configure Model*. Multiple model configurations have been implemented (e.g. XGBoost, Random Forest, Ensemble Models), supporting different model types and training strategies with resampled synthesised training data. 

4. *Perform hyperparameter tuning*. Optimise model hyperparameters to maximise performance according to a chosen objective function.  This has been optimized across model types through configuration files rather than hard-coded values, enabling experimentation with different hyperparameters for different models.

