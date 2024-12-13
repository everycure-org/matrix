import pytest
import pandas as pd
from pandera.errors import SchemaError

from matrix.pipelines.preprocessing.nodes import lorem_ipsum


def test_lorem_ipsum_validation():
    # Test valid input
    valid_df = pd.DataFrame(
        {
            "ID": [1.0, 2.0, 3.0],
            "name": ["name1", "name2", "name3"],
            "curie": ["curie1", "curie2", "curie3"],
            "normalized_curie": ["norm1", "norm2", "norm3"],
        }
    )

    # This should work without raising an exception
    result = lorem_ipsum(valid_df)
    assert result.equals(valid_df)

    # Test invalid input - wrong type for ID column
    invalid_df_1 = pd.DataFrame(
        {
            "ID": ["1", "2", "3"],  # strings instead of floats
            "name": ["name1", "name2", "name3"],
            "curie": ["curie1", "curie2", "curie3"],
            "normalized_curie": ["norm1", "norm2", "norm3"],
        }
    )

    with pytest.raises(SchemaError):
        lorem_ipsum(invalid_df_1)

    # Test invalid input - missing required column
    invalid_df_2 = pd.DataFrame(
        {
            "ID": [1.0, 2.0, 3.0],
            "name": ["name1", "name2", "name3"],
            # missing "curie" column
            "normalized_curie": ["norm1", "norm2", "norm3"],
        }
    )

    with pytest.raises(SchemaError):
        lorem_ipsum(invalid_df_2)
