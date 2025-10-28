import errno
import logging
import os

import polars as pl
import pyarrow.dataset as ds
from kedro.io.core import (
    PROTOCOL_DELIMITER,
    DatasetError,
    get_filepath_str,
)
from kedro_datasets.polars.csv_dataset import CSVDataset
from kedro_datasets.polars.lazy_polars_dataset import LazyPolarsDataset

logger = logging.Logger(__name__)


class LazyPolarsTSVDataset(LazyPolarsDataset):
    def load(self) -> pl.LazyFrame:
        load_path = str(self._get_load_path())
        if not self._exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), load_path)

        if self._protocol == "file":
            # With local filesystems, we can use Polar's build-in I/O method:
            load_method = getattr(pl, f"scan_{self._file_format}", None)
            return load_method(load_path, **self._load_args)  # type: ignore[misc]

        # For object storage, we use pyarrow for I/O:
        dataset = ds.dataset(load_path, filesystem=self._fs, format=self._file_format, **self._load_args)
        return pl.scan_pyarrow_dataset(dataset)

    def save(self, data: pl.DataFrame | pl.LazyFrame) -> None:
        save_path = get_filepath_str(self._get_save_path(), self._protocol)

        collected_data = None
        if isinstance(data, pl.LazyFrame):
            collected_data = data.collect()
        else:
            collected_data = data

        # Note: polars does support writing partitioned parquet file
        # it is leveraging Arrow to do so, see e.g.
        # https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.DataFrame.write_parquet.html
        save_method = getattr(collected_data, f"write_{self._file_format}", None)
        if save_method:
            with self._fs.open(save_path, **self._fs_open_args_save) as fs_file:
                save_method(file=fs_file, **self._save_args)

                self._invalidate_cache()
        # How the LazyPolarsDataset logic is currently written with
        # ACCEPTED_FILE_FORMATS and a check in the `__init__` method,
        # this else loop is never reached, hence we exclude it from coverage report
        # but leave it in for consistency between the Eager and Lazy classes
        else:  # pragma: no cover
            raise DatasetError(
                f"Unable to retrieve 'polars.DataFrame.write_{self._file_format}' "
                "method, please ensure that your 'file_format' parameter has been "
                "defined correctly as per the Polars API"
                "https://pola-rs.github.io/polars/py-polars/html/reference/dataframe/index.html"
            )


class CSVLazyDataset(CSVDataset):
    def load(self) -> pl.LazyFrame:
        load_path = str(self._get_load_path())
        if self._protocol == "file":
            return pl.scan_csv(load_path, **self._load_args)

        load_path = f"{self._protocol}{PROTOCOL_DELIMITER}{load_path}"
        print(**self._load_args)
        return pl.scan_csv(load_path, storage_options=self._storage_options, **self._load_args)

    def save(self, data: pl.LazyFrame) -> None:
        save_path = get_filepath_str(self._get_save_path(), self._protocol)

        with self._fs.open(save_path, **self._fs_open_args_save) as fs_file:
            # data.write_csv(file=fs_file, **self._save_args)
            data.sink_csv(path=fs_file, **self._save_args)

        self._invalidate_cache()
