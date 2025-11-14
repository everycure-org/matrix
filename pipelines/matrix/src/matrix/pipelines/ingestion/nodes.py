import pandas as pd
import pandera.pandas as pa


# FUTURE: make schema checks dynamic per transform function
@pa.check_output(
    pa.DataFrameSchema(
        columns={
            "id": pa.Column(
                str,
                nullable=False,
                checks=[
                    pa.Check(
                        lambda col: len(col.unique()) == len(col),
                        title="id must be unique",
                    )
                ],
            ),
            "translator_id": pa.Column(
                str,
                nullable=False,
                checks=[
                    pa.Check(
                        lambda col: len(col.unique()) == len(col),
                        title="translator_id must be unique",
                    )
                ],
            ),
            "deleted": pa.Column(bool, nullable=False, checks=[pa.Check(lambda col: col == False)]),
        },
    )
)
def write_drug_list(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["deleted"]]


@pa.check_output(
    pa.DataFrameSchema(
        columns={
            "id": pa.Column(
                str,
                nullable=False,
                checks=[
                    pa.Check(
                        lambda col: len(col.unique()) == len(col),
                        title="id must be unique",
                    )
                ],
            ),
            "deleted": pa.Column(bool, nullable=False, checks=[pa.Check(lambda col: col == False)]),
        },
    )
)
def write_disease_list(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["deleted"]]
