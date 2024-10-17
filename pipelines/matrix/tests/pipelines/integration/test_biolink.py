import pandas as pd

from matrix.pipelines.integration import biolink


def test_unnest():
    # Given an input dictionary of hierarchical predicate definition
    predicates = [
        {
            "name": "related_to",
            "children": [
                {"name": "composed_primarily_of", "parent": "related_to"},
                {
                    "name": "related_to_at_concept_level",
                    "parent": "related_to",
                    "children": [
                        {"name": "broad_match", "parent": "related_to_at_concept_level"},
                    ],
                },
            ],
        }
    ]

    # When calling the unnest function
    result = biolink.unnest(predicates, parents=[])
    expected = pd.DataFrame(
        [
            ["composed_primarily_of", ["related_to"], 2],
            ["broad_match", ["related_to", "related_to_at_concept_level"], 3],
            ["related_to_at_concept_level", ["related_to"], 2],
            ["related_to", [], 1],
        ],
        columns=["name", "parents", "depth"],
    )

    # Then correct mapping returned
    assert result.equals(expected)
