import pandas as pd

from sklearn.preprocessing import FunctionTransformer


class FlatArrayTransformer(FunctionTransformer):
    """sklearn compatible transformer to flatten arrays in a dataframe column.

    WARNING: Currently only supports a single input column.
    """

    def __init__(self, prefix: str):
        self.prefix = prefix
        super().__init__(self.flatten_df_rows)

    @staticmethod
    def flatten_df_rows(df):
        return pd.DataFrame(df[df.columns[0]].tolist()).to_numpy()

    def get_feature_names_out(self, input_features=None):
        if input_features.shape[1] > 1:
            raise ValueError("Only one input column is supported.")

        return [f"{self.prefix}{i}" for i in range(len(input_features.iloc[0][0]))]


df = pd.DataFrame([[[1, 2, 3], "a"], [[3, 4, 5], "b"]], columns=["embedding", "label"])

transform = FlatArrayTransformer(prefix="embedding_")

features = ["embedding"]
features_selected = df[features]
# transformer = transform.fit(df[features], None)

print(transform.transform(features_selected))

transformed = pd.DataFrame(
    transform.transform(features_selected),
    index=features_selected.index,
    columns=transform.get_feature_names_out(features_selected),
)

df = pd.concat([df.drop(columns=features), transformed], axis="columns")

print(df)
