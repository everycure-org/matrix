The matrix generation pipeline scores all drug-disease pairs using trained models. The process includes flags for known positives and negatives, exclusion of training data for unbiased scoring during evaluation. 

Note that the matrix generation is performed for each fold, as well as the "full split", where the entirety of available ground truth is used for training.

The matrix generation pipeline also outputs several plots and tables giving a global description of the output predictions. 