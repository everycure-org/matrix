import subprocess
from typing import Any

import pandas as pd
from kedro_datasets.pandas import GBQTableDataset as KedroGBQTableDataset


class GBQTableDataset(KedroGBQTableDataset):
    """
    A Kedro dataset for loading and saving pandas DataFrames to/from Google BigQuery tables.

    This dataset extends the base GBQTableDataset with the ability to add labels to tables
    """

    def __init__(
        self,
        project: str,
        dataset: str,
        table_name: str,
        credentials: Any | None = None,
        load_args: dict[str, Any] | None = None,
        save_args: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        label_key: str | None = None,
        label_value: str | None = None,
    ) -> None:
        """
        Initialize the Custom GBQ Table Dataset.

        Args:
            project: Google Cloud project ID
            dataset: BigQuery dataset name
            table_name: BigQuery table name
            credentials: Google Cloud credentials (optional)
            load_args: Additional arguments passed to pandas.read_gbq()
            save_args: Additional arguments passed to pandas.DataFrame.to_gbq()
            metadata: Arbitrary metadata for the dataset
            label_key: Key for the table label to set after saving (optional)
            label_value: Value for the table label to set after saving (optional)
        """
        super().__init__(
            project=project,
            dataset=dataset,
            table_name=table_name,
            credentials=credentials,
            load_args=load_args,
            save_args=save_args,
            metadata=metadata,
        )
        self.project = project
        self.dataset = dataset
        self.table_name = table_name
        self.label_key = label_key
        self.label_value = label_value

    def save(self, data: pd.DataFrame) -> None:
        super().save(data)

        # Set table label if both key and value are provided
        if self.label_key and self.label_value:
            table_ref = f"{self.project}:{self.dataset}.{self.table_name}"

            command = [
                "bq",
                "update",
                "--set_label",
                f"{self.label_key}:{self.label_value}",
                table_ref,
            ]

            subprocess.run(command, check=True)
