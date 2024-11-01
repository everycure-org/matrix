import requests

import pandas as pd

from typing import Callable, List, Dict, Tuple
from functools import partial

from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.inline_primary_key import primary_key

from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.schema import HumanMessage, SystemMessage
from refit.v1.core.inject import inject_object

from matrix.pipelines.integration.nodes import batch_map_ids


def resolve(name: str, endpoint: str, att_to_get: str = "preferred_curie") -> str:
    """Function to retrieve curie through the synonymizer.

    Args:
        name: name of the node
        endpoint: endpoint of the synonymizer
        att_to_get: attribute to get from API
    Returns:
        Corresponding curie
    """
    result = requests.get(f"{endpoint}/synonymize", json={"names": [name]})
    element = result.json().get(name)
    if element:
        return element.get(att_to_get, None)

    return None


def normalize(curie: str, endpoint: str, att_to_get: str = "identifier"):
    """Function to retrieve the normalized identifier through the normalizer.

    Args:
        curie: curie of the node
        endpoint: endpoint of the synonymizer
        att_to_get: attribute to get from API
    Returns:
        Corresponding curie
    """
    if not curie or pd.isna(curie):
        return None
    result = requests.get(f"{endpoint}/normalize", json={"names": [curie]})
    element = result.json().get(curie)
    if element:
        return element.get("id", {}).get(att_to_get)

    return None


def coalesce(s: pd.Series, *series: List[pd.Series]):
    """Coalesce the column information like a SQL coalesce."""
    for other in series:
        s = s.mask(pd.isnull, other)
    return s


def enrich_df(df: pd.DataFrame, endpoint: str, func: Callable, input_cols: str, target_col: str) -> pd.DataFrame:
    """Function to resolve nodes of the nodes input dataset.

    Args:
        df: nodes dataframe
        endpoint: endpoint of the synonymizer
        func: func to call
        input_cols: input cols, cols are coalesced to obtain single column
        target_col: target col
    Returns:
        dataframe enriched with Curie column
    """
    # Coalesce input cols
    col = coalesce(*[df[col] for col in input_cols])

    # Apply enrich function and replace nans by empty space
    df[target_col] = col.apply(partial(func, endpoint=endpoint))

    return df


@has_schema(
    schema={
        "ID": "numeric",
        "name": "object",
        "curie": "object",
        "normalized_curie": "object",
    },
    allow_subset=True,
)
@primary_key(primary_key=["ID"])
def create_int_nodes(nodes: pd.DataFrame, arax_endpoint: str, translator_endpoint: str) -> pd.DataFrame:
    """Function to create a intermediate nodes dataset by filtering and renaming columns."""
    # Enrich curie with node synonymizer
    resolved = enrich_df(nodes, arax_endpoint, resolve, input_cols=["name"], target_col="curie")

    # Normalize curie, by taking corrected currie or curie
    normalized_id_map = batch_map_ids(
        frozenset(resolved["curie"].fillna("")),
        api_endpoint=translator_endpoint,
        batch_size=1000,
        parallelism=120,
        conflate=True,
        drug_chemical_conflate=False,
        att_to_get="identifier",
    )
    resolved["normalized_curie"] = resolved["curie"].map(normalized_id_map)

    # If new id is specified, we use the new id as a new KG identifier should be introduced
    resolved["normalized_curie"] = coalesce(resolved["new_id"], resolved["normalized_curie"])

    return resolved


@has_schema(
    schema={
        "SourceId": "object",
        "TargetId": "object",
    },
    allow_subset=True,
)
def create_int_edges(int_nodes: pd.DataFrame, int_edges: pd.DataFrame) -> pd.DataFrame:
    """Function to create int edges dataset.

    Function ensures edges dataset link curies in the KG.
    """
    # Remove all nodes that could not be resolved, as we wont include
    # any edges between those.
    index = int_nodes[int_nodes["normalized_curie"].notna()]

    res = (
        int_edges.merge(
            index.rename(columns={"normalized_curie": "SourceId"}),
            left_on="Source",
            right_on="ID",
            how="left",
        )
        .drop(columns="ID")
        .merge(
            index.rename(columns={"normalized_curie": "TargetId"}),
            left_on="Target",
            right_on="ID",
            how="left",
        )
        .drop(columns="ID")
    )

    res["Included"] = res.apply(lambda row: not (pd.isna(row["SourceId"]) or pd.isna(row["TargetId"])), axis=1)

    return res


@has_schema(
    schema={
        "category": "object",
        "id": "object",
        "name": "object",
        "description": "object",
    },
    allow_subset=True,
)
@primary_key(primary_key=["id"])
def create_prm_nodes(prm_nodes: pd.DataFrame) -> pd.DataFrame:
    """Function to create a primary nodes that contains only new nodes introduced by the source."""
    # `new_id` signals that the node should be added to the KG as a new id
    # we drop the original ID from the spreadsheat, and leverage the new_id as the final id
    # in the dataframe. We only retain nodes where the new_id is set
    res = prm_nodes[prm_nodes["new_id"].notna()].drop(columns="ID").rename(columns={"new_id": "id"})

    res["category"] = "biolink:" + prm_nodes["entity label"]

    return res


@has_schema(
    schema={
        "subject": "object",
        "predicate": "object",
        "object": "object",
        "knowledge_source": "object",
    },
    allow_subset=True,
)
@primary_key(primary_key=["subject", "predicate", "object"])
def create_prm_edges(int_edges: pd.DataFrame) -> pd.DataFrame:
    """Function to create a primary edges dataset by filtering and renaming columns."""
    # Replace empty strings with nan
    res = int_edges.rename(columns={"SourceId": "subject", "TargetId": "object", "Label": "predicate"}).dropna(
        subset=["subject", "object"]
    )

    res["predicate"] = "biolink:" + res["predicate"]
    res["knowledge_source"] = "ec:medical"

    return res


@has_schema(
    schema={
        "clinical_trial_id": "object",
        "reason_for_rejection": "object",
        "drug_name": "object",
        "disease_name": "object",
        "significantly_better": "numeric",
        "non_significantly_better": "numeric",
        "non_significantly_worse": "numeric",
        "significantly_worse": "numeric",
    },
    allow_subset=True,
    df="df",
)
def map_name_to_curie(
    df: pd.DataFrame, arax_endpoint: str, translator_endpoint: str, drug_types: List[str], disease_types: List[str]
) -> pd.DataFrame:
    """Map drug name to curie.

    Function to map drug name or disease name in raw clinical trail dataset to curie using the synonymizer.
    And check after mapping, if the mapped curies are the same in different rows, we check whether their the
    clinical performance is the same. If not, we label them as "True" in the "conflict" column, otherwise "False".

    Args:
        df: raw clinical trial dataset from medical team
        arax_endpoint: endpoint of the synonymizer
        translator_endpoint: endpoint of the normalizer
        drug_types: list of drug types
        disease_types: list of disease types
    Returns:
        dataframe with two additional columns: "Mapped Drug Curie" and "Mapped Drug Disease"
    """
    # Map the drug name to the corresponding arax curie ids which we can then use by translator normalizer
    df["drug_kg_arax_curie"] = df["drug_name"].apply(lambda x: normalize(x, endpoint=arax_endpoint))
    df["disease_kg_arax_curie"] = df["disease_name"].apply(lambda x: normalize(x, endpoint=arax_endpoint))
    print(df["drug_kg_arax_curie"])
    print(df["disease_kg_arax_curie"])
    # Map the disease name to the corresponding curie ids
    attributes = [
        ("identifier", "drug_kg_curie"),
        ("label", "drug_kg_label"),
    ]

    for att, target in attributes:
        node_id_map = batch_map_ids(
            frozenset(df["drug_kg_arax_curie"].fillna("none")),
            api_endpoint=translator_endpoint,
            batch_size=1000,
            parallelism=120,
            conflate=True,
            drug_chemical_conflate=False,
            att_to_get=att,
        )
        df[target] = df["drug_kg_arax_curie"].map(node_id_map)

    attributes = [
        ("identifier", "disease_kg_curie"),
        ("label", "disease_kg_label"),
    ]

    for att, target in attributes:
        node_id_map = batch_map_ids(
            frozenset(df["disease_kg_arax_curie"].fillna("none")),
            api_endpoint=translator_endpoint,
            batch_size=1000,
            parallelism=120,
            conflate=True,
            drug_chemical_conflate=False,
            att_to_get=att,
        )
        df[target] = df["disease_kg_arax_curie"].map(node_id_map)

    # Validate correct labels
    # NOTE: This is a temp. solution that ensures clinical trails data
    # only passes on data as containend by our pre-filtering in the modelling pipeline
    # we aim to refine our evaluation approach as part of a new PR after which
    # this can be removed.
    # https://github.com/everycure-org/matrix/issues/313
    df["label_included"] = (df["drug_kg_label"].isin(drug_types)) & (df["disease_kg_label"].isin(disease_types))

    # check conflict
    df["conflict"] = (
        df.groupby(["drug_kg_curie", "disease_kg_curie"])[
            [
                "significantly_better",
                "non_significantly_better",
                "non_significantly_worse",
                "significantly_worse",
            ]
        ]
        .transform(lambda x: x.nunique() > 1)
        .any(axis=1)
    )

    return df


@has_schema(
    schema={
        "clinical_trial_id": "object",
        "reason_for_rejection": "object",
        "drug_name": "object",
        "disease_name": "object",
        "drug_kg_curie": "object",
        "disease_kg_curie": "object",
        "conflict": "object",
        "significantly_better": "numeric",
        "non_significantly_better": "numeric",
        "non_significantly_worse": "numeric",
        "significantly_worse": "numeric",
    },
    allow_subset=True,
    df="df",
)
@primary_key(
    primary_key=[
        "clinical_trial_id",
        "drug_kg_curie",
        "disease_kg_curie",
    ]
)
def clean_clinical_trial_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean clinical trails data.

    Function to clean the mapped clinical trial dataset for use in time-split evaluation metrics.

    Args:
        df: raw clinical trial dataset added with mapped drug and disease curies
    Returns:
        Cleaned clinical trial data.
    """
    # Remove rows with conflicts
    df = df[df["conflict"].eq("FALSE")].reset_index(drop=True)

    # Make sure to consider only rows with relevant labels, otherwise
    # downtstream modelling will fail
    df = df[df["label_included"].eq("TRUE")].reset_index(drop=True)

    # remove rows with reason for rejection
    df = df[df["reason_for_rejection"].isna()].reset_index(drop=True)

    # Define columns to check
    columns_to_check = [
        "drug_kg_curie",
        "disease_kg_curie",
        "significantly_better",
        "non_significantly_better",
        "non_significantly_worse",
        "significantly_worse",
    ]

    # Remove rows with missing values in cols
    df = df.dropna(subset=columns_to_check).reset_index(drop=True)

    # drop columns
    df = df.drop(columns=["reason_for_rejection", "conflict"]).reset_index(drop=True)

    return df


@has_schema(
    schema={"single_ID": "object", "curie": "object", "name": "object"},
    allow_subset=True,
)
# @primary_key(primary_key=["single_ID"]) #TODO: re-introduce once the drug list is ready
def clean_drug_list(drug_df: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    """Synonymize the drug list and filter out NaNs.

    Args:
        drug_df: disease list in a dataframe format.
        endpoint: endpoint of the synonymizer.

    Returns:
        dataframe with synonymized drug IDs in normalized_curie column.
    """
    attributes = [
        ("identifier", "curie"),
        ("label", "name"),
        ("type", "category"),
    ]

    for att, target in attributes:
        node_id_map = batch_map_ids(
            frozenset(drug_df["single_ID"]),
            api_endpoint=endpoint,
            batch_size=1000,
            parallelism=120,
            conflate=True,
            drug_chemical_conflate=False,
            att_to_get=att,
        )
        drug_df[target] = drug_df["single_ID"].map(node_id_map)
    return drug_df.dropna(subset=["curie"])


@has_schema(
    schema={
        "category_class": "object",
        "label": "object",
        "definition": "object",
        "synonyms": "object",
        "subsets": "object",
        "crossreferences": "object",
        "curie": "object",
        "name": "object",
    },
    allow_subset=True,
)
@primary_key(primary_key=["category_class", "curie"])
def clean_disease_list(disease_df: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    """Synonymize the IDs, names, and categories within disease list and filter out NaNs.

    Args:
        disease_df: disease list in a dataframe format.
        endpoint: endpoint of the synonymizer.

    Returns:
        dataframe with synonymized disease IDs in normalized_curie column.
    """
    attributes = [
        ("identifier", "curie"),
        ("label", "name"),
        ("type", "category"),
    ]

    for att, target in attributes:
        node_id_map = batch_map_ids(
            frozenset(disease_df["category_class"]),
            api_endpoint=endpoint,
            batch_size=1000,
            parallelism=120,
            conflate=True,
            drug_chemical_conflate=False,
            att_to_get=att,
        )
        disease_df[target] = disease_df["category_class"].map(node_id_map)
    return disease_df.dropna(subset=["curie"]).fillna("")


@has_schema(
    {
        "timestamp": "object",
        "drug_id": "object",
        "disease_id": "object",
        "norm_drug_id": "object",
        "norm_disease_id": "object",
        "norm_drug_name": "object",
        "norm_disease_name": "object",
    },
    allow_subset=True,
)
def clean_input_sheet(input_df: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    """Synonymize the input sheet and filter out NaNs.

    Args:
        input_df: input list in a dataframe format.
        endpoint: endpoint of the synonymizer.

    Returns:
        dataframe with synonymized disease IDs in normalized_curie column.
    """
    # Synonymize Drug_ID column to normalized ID and name compatible with RTX-KG2
    attributes = [
        ("identifier", "norm_drug_id"),
        ("label", "norm_drug_name"),
    ]

    for att, target in attributes:
        node_id_map = batch_map_ids(
            frozenset(input_df["Drug_ID"]),
            api_endpoint=endpoint,
            batch_size=1000,
            parallelism=120,
            conflate=True,
            drug_chemical_conflate=False,
            att_to_get=att,
        )
        input_df[target] = input_df["Drug_ID"].map(node_id_map)

    for att, target in attributes:
        node_id_map = batch_map_ids(
            frozenset(input_df["Disease_ID"]),
            api_endpoint=endpoint,
            batch_size=1000,
            parallelism=120,
            conflate=True,
            drug_chemical_conflate=False,
            att_to_get=att,
        )
        input_df[target] = input_df["Disease_ID"].map(node_id_map)

    # Select columns of interest and rename
    col_list = [
        "Timestamp",
        "Drug_ID",
        "Disease_ID",
        "norm_drug_id",
        "norm_drug_name",
        "norm_disease_id",
        "norm_disease_name",
    ]
    df = input_df.loc[:, col_list]
    df.columns = [string.lower() for string in col_list]

    # Fill NaNs and return
    return df.fillna("")


# GT


def clean_gt_data(pos_df: pd.DataFrame, neg_df: pd.DataFrame, endpoint: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Clean ground truth data.

    Args:
        pos_df: positive ground truth data.
        neg_df: negative ground truth data.
        endpoint: endpoint of the synonymizer.
    Returns:
        Cleaned ground truth data.
    """
    # Synonymize source and target IDs for both positive and negative ground truth data
    for df in [pos_df, neg_df]:
        for col in ["source", "target"]:
            node_id_map = batch_map_ids(
                frozenset(df[col]),
                api_endpoint=endpoint,
                batch_size=1000,
                parallelism=120,
                conflate=True,
                drug_chemical_conflate=False,
                att_to_get="identifier",
            )
            df[col] = df[col].map(node_id_map)

    # Return updated DataFrames
    return pos_df.dropna(subset=["source", "target"]).drop_duplicates(), neg_df.dropna(
        subset=["source", "target"]
    ).drop_duplicates()


# FUTURE: Remove the functions once we have tags embedded in the disease list
@inject_object()
def generate_tag(
    disease_list: List, model: Dict, definitions: str = None, synonyms: str = None, raw_prompt: str = None
) -> List:
    """Temporary function to generate tags based on provided prompts and params through OpenAI API call.

    This function is temporary and will be removed once we have tags embedded in the disease list.

    Args:
        disease_list: list- list of disease for which tags should be generated.
        definitions: str - (optional) definition of the disease, needed for prompts requiring multiple inputs.
        synonyms: str - (optional) synonyms of the disease, needed for prompts requiring multiple inputs.
        prompt: str - prompt for the tag generation
        llm_model: str - name of the llm model to use for tag generation
    Returns
        List of tags generated by the API call.
    """
    # Initialize the output parser
    output_parser = CommaSeparatedListOutputParser()

    # Generate tags
    tag_list = []
    for i, disease in enumerate(disease_list):
        if (definitions is None) | (synonyms is None):
            prompt = ChatPromptTemplate.from_messages(
                [SystemMessage(content=raw_prompt), HumanMessage(content=disease)]
            )
            formatted_prompt = prompt.format_messages(disease=disease)
        else:
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content=raw_prompt.format(disease=disease, synonym=synonyms[i], definition=definitions[i])
                    )
                ]
            )

            formatted_prompt = prompt.format_messages(disease=disease, synonym=synonyms[i], definition=definitions[i])
        response = model.invoke(formatted_prompt)
        tags = output_parser.parse(response.content)
        tag_list.append(", ".join(tags))
    return tag_list


def enrich_disease_list(disease_list: List, params: Dict) -> pd.DataFrame:
    """Temporary function to enrich existing disease list with llm-generated tags.

    This function  will be removed once we have tags embedded in the disease list.

    Args:
        disease_list: pd.DataFrame - merged disease_list with disease names column that will be used for tag generation
        params: Dict - parameters dictionary specifying tag names, column names, and model params
        llm_model:  - name of the llm model to use for tag generation
    Returns
        pd.DataFrame with x new tag columns (where x corresponds to number of tags specified in params)
    """
    disease_list = disease_list
    for input_type in ["single_input", "multiple_input"]:
        input_params = params[input_type]
        for tag, tag_params in input_params.items():
            # Check if tag is already in disease list
            if tag in disease_list.columns:
                continue

            print(f"Applying tag: '{tag}' to disease list")

            input_col = tag_params["input_params"]["input_col"]
            output_col = tag_params["input_params"]["output_col"]
            raw_prompt = tag_params["input_params"]["prompt"]
            model = tag_params["model_params"]

            # Check whether the tag needs a single or multiple inputs
            if input_type == "single_input":
                disease_list[output_col] = generate_tag(
                    disease_list=disease_list[input_col], raw_prompt=raw_prompt, model=model
                )
            else:
                definition_col = tag_params["input_params"]["definition"]
                synonym_col = tag_params["input_params"]["synonyms"]
                disease_list[output_col] = generate_tag(
                    disease_list=disease_list[input_col],
                    definitions=disease_list[definition_col],
                    synonyms=disease_list[synonym_col],
                    raw_prompt=raw_prompt,
                    model=model,
                )
    return disease_list
