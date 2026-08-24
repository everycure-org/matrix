import ast
import asyncio
import logging

import nest_asyncio
import numpy as np
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
            "synonyms": pa.Column(nullable=True),
        },
        unique=["category_class"],
    )
)
@pa.check_output(
    pa.DataFrameSchema(
        columns={
            "id": pa.Column(nullable=False),
            "name": pa.Column(nullable=False),
            "synonyms": pa.Column(dtype=list[str], nullable=False),
        },
        unique=["id"],
        strict=True,
    )
)
def ingest_source_disease_list(disease_list: pd.DataFrame) -> pd.DataFrame:
    disease_list["synonyms"] = disease_list.synonyms.apply(
        lambda x: [] if pd.isna(x) else [xx.strip() for xx in x.split(";") if xx.strip() != ""]
    )
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


def split_new_diseases(disease_list: pd.DataFrame, previous_output: pd.DataFrame) -> pd.DataFrame:
    """Keep only diseases that are not already present in a previous LLM output, so the LLM graph only runs on new diseases."""
    previously_processed_ids = set(previous_output["id"])
    new_diseases = disease_list[~disease_list["id"].isin(previously_processed_ids)]
    logger.info(
        f"{len(new_diseases)} new diseases out of {len(disease_list)} will be processed by the LLM graph "
        f"({len(disease_list) - len(new_diseases)} reused from previous output)"
    )
    return new_diseases.reset_index(drop=True)


def parse_boolean(b: any) -> bool:
    if b is None or pd.isna(b) or b == "":
        return None
    elif isinstance(b, bool):
        return b
    elif isinstance(b, str):
        return b.lower() == "true"
    else:
        raise ValueError(f"Invalid boolean value: {b}")


def merge_with_categories_previous_output(
    new_output: pd.DataFrame, previous_output: pd.DataFrame, disease_list: pd.DataFrame
) -> pd.DataFrame:
    """Combine freshly computed LLM output with reused rows from a previous output, for diseases still in the current list."""
    reused_output = previous_output[previous_output["id"].isin(disease_list["id"])]

    if not new_output.empty:
        formatted_new_output = new_output[reused_output.columns]
    else:
        formatted_new_output = pd.DataFrame(columns=reused_output.columns)

    for column in reused_output.columns:
        if column in ["id", "name"]:
            continue
        formatted_new_output[column] = formatted_new_output[column].astype(object)
        formatted_new_output[column] = formatted_new_output[column].apply(parse_boolean)
        reused_output[column] = reused_output[column].astype(object)
        reused_output[column] = reused_output[column].apply(parse_boolean)

    result = pd.concat([reused_output, formatted_new_output], ignore_index=True)
    return result


def _normalize_umn_synonyms(value) -> list | None:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in {"", "''", '""'}:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        if isinstance(parsed, str) and parsed.strip():
            return [parsed.strip()]
        if ";" in stripped:
            return [part.strip() for part in stripped.split(";") if part.strip()]
        return [stripped]
    raise ValueError(f"Expected list-like synonyms value, got {type(value)!r}")


def _token_counter_to_dict(value) -> dict | None:
    """Normalize LLM token counters to dict form for parquet (matches invoke_graph output)."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        try:
            value = ast.literal_eval(stripped)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Invalid token counter string: {value!r}") from exc
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Expected token counter list or dict, got {type(value)!r}")
    llm_calls = list(value)
    if len(llm_calls) == 0:
        return {}
    if isinstance(llm_calls[0], list):
        return {f"{name}_{i}": tokens for i, llm_call in enumerate(llm_calls) for name, tokens in llm_call}
    return {name: tokens for name, tokens in llm_calls}


def _align_umn_output_for_concat(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize UMN column dtypes so reused and new rows can be concatenated and written to parquet."""
    # NOTE: This function was partially generated using AI assistance.
    aligned = dataframe.copy()

    if "synonyms" in aligned.columns:
        aligned["synonyms"] = aligned["synonyms"].apply(_normalize_umn_synonyms)

    for column in aligned.columns:
        if column in {"id", "entity", "name", "synonyms", "synonym_prompt"}:
            continue
        if column.endswith("_explanation"):
            aligned[column] = aligned[column].astype("string")
        elif column in {"request_token_counter", "response_token_counter"}:
            aligned[column] = aligned[column].apply(_token_counter_to_dict)
        elif column == "umn_score":
            aligned[column] = pd.to_numeric(aligned[column], errors="coerce").astype("float64")
        else:
            aligned[column] = pd.to_numeric(aligned[column], errors="coerce").astype("Int64")

    return aligned


def merge_with_umn_previous_output(
    new_output: pd.DataFrame, previous_output: pd.DataFrame, disease_list: pd.DataFrame
) -> pd.DataFrame:
    """Combine freshly computed LLM output with reused rows from a previous output, for diseases still in the current list."""
    reused_output = previous_output[previous_output["id"].isin(disease_list["id"])].copy()

    if not new_output.empty:
        formatted_new_output = new_output[reused_output.columns].copy()
    else:
        formatted_new_output = pd.DataFrame(columns=reused_output.columns)

    return pd.concat(
        [
            _align_umn_output_for_concat(reused_output),
            _align_umn_output_for_concat(formatted_new_output),
        ],
        ignore_index=True,
    )


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
            return _token_counter_to_dict(llm_calls)
        except Exception as e:
            logger.error(f"Error in move_tokens_from_tuple_to_dict: {str(e)}")
            return None

    try:
        llm_output["request_token_counter"] = llm_output["request_token_counter"].apply(move_tokens_from_tuple_to_dict)
        llm_output["response_token_counter"] = llm_output["response_token_counter"].apply(
            move_tokens_from_tuple_to_dict
        )
    except Exception as e:
        logger.error(f"Error in move_tokens_from_tuple_to_dict: {str(e)}")
        return llm_output
    return llm_output
