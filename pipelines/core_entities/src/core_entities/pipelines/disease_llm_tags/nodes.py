import asyncio
import logging

import nest_asyncio
import pandas as pd
import pandera.pandas as pa
from matrix_inject.inject import inject_object
from tqdm.asyncio import tqdm

from core_entities.utils.curation_utils import _log_merge_statistics, apply_patch
from core_entities.utils.llm_utils import InvokableGraph

logger = logging.getLogger(__name__)

nest_asyncio.apply()


@pa.check_input(
    pa.DataFrameSchema(
        columns={
            "category_class": pa.Column(nullable=False),
            "label": pa.Column(nullable=False),
            "synonyms": pa.Column(nullable=False),
        },
        unique=["category_class"],
    )
)
@pa.check_output(
    pa.DataFrameSchema(
        columns={
            "id": pa.Column(nullable=False),
            "name": pa.Column(nullable=False),
            "synonyms": pa.Column(nullable=False),
        },
        unique=["id"],
        strict=True,
    )
)
def ingest_source_disease_list(disease_list: pd.DataFrame) -> pd.DataFrame:
    return disease_list.rename(columns={"category_class": "id", "label": "name"})[["id", "name", "synonyms"]]


@pa.check_input(
    pa.DataFrameSchema(
        parsers=pa.Parser(
            lambda df: df[
                [
                    "mondo_id",
                    "level",
                ]
            ]
        ),
        columns={
            "mondo_id": pa.Column(
                nullable=False,
                checks=pa.Check(
                    lambda col: col.apply(lambda x: x.startswith("MONDO:")),
                    title="mondo_id does not start with 'MONDO:'",
                ),
            ),
            "level": pa.Column(
                nullable=True,
                checks=pa.Check(
                    lambda col: col.apply(
                        lambda x: x.strip() == "" or x in ["clinically_recognized", "subgroup", "exclude", "grouping"]
                    ),
                    ignore_na=False,
                    title="level value is valid",
                ),
            ),
        },
        unique=["mondo_id"],
    )
)
@pa.check_output(
    pa.DataFrameSchema(
        columns={
            "id": pa.Column(nullable=False),
            "level": pa.Column(
                nullable=False,
                checks=pa.Check(
                    lambda col: col.apply(lambda x: x == "clinically_recognized"),
                    title="Only clinically recognized diseases",
                ),
            ),
        },
        unique=["id"],
        strict=True,
    )
)
def ingest_curated_disease_list(curated_disease_list: pd.DataFrame) -> pd.DataFrame:
    filtered_curated_disease_list = curated_disease_list[curated_disease_list["level"] == "clinically_recognized"]
    return filtered_curated_disease_list.rename(columns={"mondo_id": "id"})


@pa.check_input(
    pa.DataFrameSchema(
        parsers=pa.Parser(
            lambda df: df[
                [
                    "id",
                    "disease_label",
                    "disease_label_explanation",
                ]
            ]
        ),
        columns={
            "id": pa.Column(nullable=False),
            "disease_label": pa.Column(nullable=False),
            "disease_label_explanation": pa.Column(nullable=False),
        },
        unique=["id"],
    )
)
@pa.check_output(
    pa.DataFrameSchema(
        columns={
            "id": pa.Column(nullable=False),
            "disease_label": pa.Column(nullable=False),
            "disease_label_explanation": pa.Column(nullable=False),
        },
        unique=["id"],
        strict=True,
    )
)
def ingest_disease_labels(disease_labels: pd.DataFrame) -> pd.DataFrame:
    return disease_labels


@pa.check_output(
    pa.DataFrameSchema(
        columns={
            "id": pa.Column(nullable=False),
            "name": pa.Column(nullable=False),
            "synonyms": pa.Column(nullable=False),
        },
        unique=["id"],
        strict=True,
    )
)
def merge_disease_lists(disease_list: pd.DataFrame, curated_disease_list: pd.DataFrame) -> pd.DataFrame:
    _log_merge_statistics(
        primary_df=disease_list,
        secondary_df=curated_disease_list,
        primary_name="disease list",
        secondary_name="curated disease list",
        primary_only_action="will be dropped",
        secondary_only_action="will be dropped",
    )
    merged_disease_list = pd.merge(disease_list, curated_disease_list, on="id", how="inner")

    return merged_disease_list.drop(columns=["level"])


def merge_disease_list_with_labels(disease_list: pd.DataFrame, disease_labels: pd.DataFrame) -> pd.DataFrame:
    _log_merge_statistics(
        primary_df=disease_list,
        secondary_df=disease_labels,
        primary_name="disease list",
        secondary_name="disease labels",
        primary_only_action="will be dropped",
        secondary_only_action="will be dropped",
    )
    merged_disease_list = pd.merge(disease_list, disease_labels, on="id", how="inner")
    return merged_disease_list


def patch_disease_name(disease_list: pd.DataFrame, disease_name_patch: pd.DataFrame) -> pd.DataFrame:
    return apply_patch(disease_list, disease_name_patch, ["name"], "id")


@pa.check_input(
    pa.DataFrameSchema(
        columns={
            "id": pa.Column(nullable=False),
        },
        unique=["id"],
    )
)
@inject_object()
def invoke_graph(
    disease_list: pd.DataFrame,
    graph: InvokableGraph,
    invoke_parameters: dict,
    parallelism: int,
    ignore_errors: bool,
) -> pd.DataFrame:
    async def invoke_for_all_rows(dataframe):
        semaphore = asyncio.Semaphore(parallelism)

        async def invoke_with_semaphore(row):
            invoke_parameters_dict = {param_name: row[param] for param_name, param in invoke_parameters.items()}
            async with semaphore:
                try:
                    graph_result = await graph.safe_invoke(**invoke_parameters_dict)
                    graph_result["id"] = row["id"]
                    return graph_result
                except Exception as e:
                    if ignore_errors:
                        logger.warning(f"Error in invoke with parameters: {invoke_parameters_dict}: {str(e)}")
                        return {"id": row["id"]}
                    else:
                        raise e

        tasks = [invoke_with_semaphore(row) for _, row in dataframe.iterrows()]

        return await tqdm.gather(*tasks)

    llm_output = pd.DataFrame(asyncio.run(invoke_for_all_rows(disease_list)))

    # Parquet doesn't like arrays of tuples with strings and integers together
    def move_tokens_from_tuple_to_dict(llm_calls):
        try:
            if len(llm_calls) > 0 and isinstance(llm_calls[0], list):
                return {f"{name}_{i}": tokens for i, llm_call in enumerate(llm_calls) for name, tokens in llm_call}
            else:
                return {f"{name}": tokens for name, tokens in llm_calls}
        except Exception as e:
            logger.error(f"Error in move_tokens_from_tuple_to_dict: {str(e)}")
            return None

    llm_output["request_token_counter"] = llm_output["request_token_counter"].apply(move_tokens_from_tuple_to_dict)
    llm_output["response_token_counter"] = llm_output["response_token_counter"].apply(move_tokens_from_tuple_to_dict)

    return llm_output
