import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

class DrugStratifiedSplit(BaseCrossValidator):
    def __init__(self, n_splits=1, test_size=0.1, random_state=None):
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        rng = np.random.RandomState(self.random_state)
        
        for iteration in range(self.n_splits):
            train_indices, test_indices = [], []

            for _, group in X.groupby('source'):
                indices = group.index.tolist()
                rng.shuffle(indices)
                n = len(indices)
                n_test = max(1, int(np.round(n * self.test_size)))
                n_train = n - n_test
                
                train_indices.extend(indices[:n_train])
                test_indices.extend(indices[n_train:])

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split_with_labels(self, data):
        all_data_frames = []
        for iteration, (train_index, test_index) in enumerate(self.split(data, data["y"])):
            fold_data = data.copy()
            fold_data.loc[:, "iteration"] = iteration
            fold_data.loc[train_index, "split"] = "TRAIN"
            fold_data.loc[test_index, "split"] = "TEST"
            all_data_frames.append(fold_data)
        return pd.concat(all_data_frames, axis="index", ignore_index=True)
