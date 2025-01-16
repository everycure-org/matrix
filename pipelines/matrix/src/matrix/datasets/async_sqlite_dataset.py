import sqlite3

from kedro.io.core import AbstractVersionedDataset as AbstractDataSet


class SQLiteDataSet(AbstractDataSet):
    """Custom dataset for reading from and writing to an SQLite database."""

    def __init__(self, table: str, db_path: str):
        self.table = table
        self.db_path = db_path

    def _load(self):
        """Load data from the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {self.table}")
        data = cursor.fetchall()
        conn.close()
        return data

    def _save(self, data):
        """Save data to the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Implement your save logic here
        conn.commit()
        conn.close()

    def _describe(self):
        return dict(table=self.table, db_path=self.db_path)
