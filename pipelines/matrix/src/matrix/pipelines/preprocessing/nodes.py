from typing import List

import pandas as pd
import requests
from refit.v1.core.inline_has_schema import has_schema
from refit.v1.core.inline_primary_key import primary_key
from tenacity import retry, wait_exponential, stop_after_attempt


def coalesce(s: pd.Series, *series: List[pd.Series]):
    """Coalesce the column information like a SQL coalesce."""
    for other in series:
        s = s.mask(pd.isnull, other)
    return s


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def resolve_name(name: str, cols_to_get: List[str]) -> dict:
    """Function to retrieve the normalized identifier through the normalizer.

    Args:
        name: name of the node to be resolved
        cols_to_get: attribute to get from API
    Returns:
        Name and corresponding curie
    """

    if not name or pd.isna(name):
        return {}

    result = requests.get(
        f"https://name-resolution-sri-dev.apps.renci.org/lookup?string={name}&autocomplete=True&highlighting=False&offset=0&limit=1"
    )
    if len(result.json()) != 0:
        element = result.json()[0]
        print({col: element.get(col) for col in cols_to_get})
        return {col: element.get(col) for col in cols_to_get}

    return {}


@has_schema(
    schema={"ID": "numeric", "name": "object", "curie": "object", "description": "object"},
)
@primary_key(primary_key=["ID"])
def process_medical_nodes(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize the name
    enriched_data = df["name"].apply(resolve_name, cols_to_get=["curie", "label", "types"])

    # Extract into df
    enriched_df = pd.DataFrame(enriched_data.tolist())
    df = pd.concat([df, enriched_df], axis=1)

    # Coalesce id and new id to allow adding "new" nodes
    df["normalized_curie"] = coalesce(df["new_id"], df["curie"])

    return df


@has_schema(
    schema={
        "SourceId": "object",
        "TargetId": "object",
    },
    allow_subset=True,
)
def process_medical_edges(int_nodes: pd.DataFrame, int_edges: pd.DataFrame) -> pd.DataFrame:
    """Function to create int edges dataset.

    Function ensures edges dataset link curies in the KG.
    """
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
        "clinical_trial_id": "object",
        "reason_for_rejection": "object",
        "drug_name": "object",
        "disease_name": "object",
        "significantly_better": "numeric",
        "non_significantly_better": "numeric",
        "non_significantly_worse": "numeric",
        "significantly_worse": "numeric",
        "conflict": "bool",
    },
    allow_subset=True,
)
def add_source_and_target_to_clinical_trails(df: pd.DataFrame) -> pd.DataFrame:
    df = df.head(10)

    # Normalize the name
    drug_data = df["drug_name"].apply(resolve_name, cols_to_get=["curie"])
    disease_data = df["disease_name"].apply(resolve_name, cols_to_get=["curie"])

    # Concat dfs
    drug_df = pd.DataFrame(drug_data.tolist()).rename(columns={"curie": "drug_curie"})
    disease_df = pd.DataFrame(disease_data.tolist()).rename(columns={"curie": "disease_curie"})
    df = pd.concat([df, drug_df, disease_df], axis=1)

    # Check values
    cols = [
        "significantly_better",
        "non_significantly_better",
        "non_significantly_worse",
        "significantly_worse",
    ]

    # check conflict
    df["conflict"] = df.groupby(["drug_curie", "disease_curie"])[cols].transform(lambda x: x.nunique() > 1).any(axis=1)

    return df


@has_schema(
    schema={
        "curie": "object",
        "name": "object",
    },
    allow_subset=True,
    output=0,
    df=None,
)
@has_schema(
    schema={
        "clinical_trial_id": "object",
        "drug_name": "object",
        "disease_name": "object",
        "drug_curie": "object",
        "disease_curie": "object",
        "significantly_better": "numeric",
        "non_significantly_better": "numeric",
        "non_significantly_worse": "numeric",
        "significantly_worse": "numeric",
    },
    allow_subset=True,
    output=1,
    df=None,
)
@primary_key(
    primary_key=[
        "clinical_trial_id",
        "drug_curie",
        "disease_curie",
    ],
    output=1,
    df=None,
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

    # remove rows with reason for rejection
    df = df[df["reason_for_rejection"].isna()].reset_index(drop=True)

    # Define columns to check
    columns_to_check = [
        "drug_curie",
        "disease_curie",
        "significantly_better",
        "non_significantly_better",
        "non_significantly_worse",
        "significantly_worse",
    ]

    # Remove rows with missing values in cols
    df = df.dropna(subset=columns_to_check).reset_index(drop=True)
    edges = df.drop(columns=["reason_for_rejection", "conflict"]).reset_index(drop=True)

    # extract nodes
    drugs = df.rename(columns={"drug_curie": "curie", "drug_name": "name"})[["curie", "name"]]
    diseases = df.rename(columns={"disease_curie": "curie", "disease_name": "name"})[["curie", "name"]]
    nodes = pd.concat([drugs, diseases], ignore_index=True)

    return [nodes, edges]


# @has_schema(
#     schema={"single_ID": "object", "curie": "object", "name": "object"},
#     allow_subset=True,
# )
# # @primary_key(primary_key=["single_ID"]) #TODO: re-introduce once the drug list is ready
# def clean_drug_list(
#     drug_df: pd.DataFrame,
#     endpoint: str,
#     conflate: bool,
#     drug_chemical_conflate: bool,
#     batch_size: int,
#     parallelism: int,
# ) -> pd.DataFrame:
#     """Synonymize the drug list and filter out NaNs.

#     Args:
#         drug_df: disease list in a dataframe format.
#         endpoint: endpoint of the synonymizer.
#         conflate: whether to conflate
#         drug_chemical_conflate: whether to conflate drug and chemical
#         batch_size: batch size
#         parallelism: parallelism
#     Returns:
#         dataframe with synonymized drug IDs in normalized_curie column.
#     """
#     attributes = [
#         ("$.id.identifier", "curie"),
#         ("$.id.label", "name"),
#         ("$.type[0]", "category"),
#     ]
#     for expr, target in attributes:
#         json_parser = parse(expr)
#         node_id_map = batch_map_ids(
#             frozenset(drug_df["single_ID"]),
#             api_endpoint=endpoint,
#             batch_size=batch_size,
#             parallelism=parallelism,
#             conflate=conflate,
#             drug_chemical_conflate=drug_chemical_conflate,
#             json_parser=json_parser,
#         )
#         drug_df[target] = drug_df["single_ID"].map(node_id_map)
#     return drug_df.dropna(subset=["curie"])


# @has_schema(
#     schema={
#         "category_class": "object",
#         "label": "object",
#         "definition": "object",
#         "synonyms": "object",
#         "subsets": "object",
#         "crossreferences": "object",
#         "curie": "object",
#         "name": "object",
#     },
#     allow_subset=True,
# )
# @primary_key(primary_key=["category_class", "curie"])
# def clean_disease_list(
#     disease_df: pd.DataFrame,
#     endpoint: str,
#     conflate: bool,
#     drug_chemical_conflate: bool,
#     batch_size: int,
#     parallelism: int,
# ) -> pd.DataFrame:
#     """Synonymize the IDs, names, and categories within disease list and filter out NaNs.

#     Args:
#         disease_df: disease list in a dataframe format.
#         endpoint: endpoint of the synonymizer.
#         conflate: whether to conflate
#         drug_chemical_conflate: whether to conflate drug and chemical
#         batch_size: batch size
#         parallelism: parallelism

#     Returns:
#         dataframe with synonymized disease IDs in normalized_curie column.
#     """
#     attributes = [
#         ("$.id.identifier", "curie"),
#         ("$.id.label", "name"),
#         ("$.type[0]", "category"),
#     ]
#     for expr, target in attributes:
#         json_parser = parse(expr)
#         node_id_map = batch_map_ids(
#             frozenset(disease_df["category_class"]),
#             api_endpoint=endpoint,
#             batch_size=batch_size,
#             parallelism=parallelism,
#             conflate=conflate,
#             drug_chemical_conflate=drug_chemical_conflate,
#             json_parser=json_parser,
#         )
#         disease_df[target] = disease_df["category_class"].map(node_id_map)
#     return disease_df.dropna(subset=["curie"]).fillna("")


# @has_schema(
#     {
#         "timestamp": "object",
#         "drug_id": "object",
#         "disease_id": "object",
#         "norm_drug_id": "object",
#         "norm_disease_id": "object",
#         "norm_drug_name": "object",
#         "norm_disease_name": "object",
#     },
#     allow_subset=True,
# )
# def clean_input_sheet(
#     input_df: pd.DataFrame,
#     endpoint: str,
#     conflate: bool,
#     drug_chemical_conflate: bool,
#     batch_size: int,
#     parallelism: int,
# ) -> pd.DataFrame:
#     """Synonymize the input sheet and filter out NaNs.

#     Args:
#         input_df: input list in a dataframe format.
#         endpoint: endpoint of the synonymizer.
#         conflate: whether to conflate
#         drug_chemical_conflate: whether to conflate drug and chemical
#         batch_size: batch size
#         parallelism: parallelism
#     Returns:
#         dataframe with synonymized disease IDs in normalized_curie column.
#     """
#     # Synonymize Drug_ID column to normalized ID and name compatible with RTX-KG2
#     attributes = [
#         ("$.id.identifier", "norm_drug_id"),
#         ("$.id.label", "norm_drug_name"),
#     ]
#     for expr, target in attributes:
#         json_parser = parse(expr)
#         node_id_map = batch_map_ids(
#             frozenset(input_df["Drug_ID"]),
#             api_endpoint=endpoint,
#             batch_size=batch_size,
#             parallelism=parallelism,
#             conflate=conflate,
#             drug_chemical_conflate=drug_chemical_conflate,
#             json_parser=json_parser,
#         )
#         input_df[target] = input_df["Drug_ID"].map(node_id_map)

#     for expr, target in attributes:
#         json_parser = parse(expr)
#         node_id_map = batch_map_ids(
#             frozenset(input_df["Disease_ID"]),
#             api_endpoint=endpoint,
#             batch_size=batch_size,
#             parallelism=parallelism,
#             conflate=conflate,
#             drug_chemical_conflate=drug_chemical_conflate,
#             json_parser=json_parser,
#         )
#         input_df[target] = input_df["Disease_ID"].map(node_id_map)

#     # Select columns of interest and rename
#     col_list = [
#         "Timestamp",
#         "Drug_ID",
#         "Disease_ID",
#         "norm_drug_id",
#         "norm_drug_name",
#         "norm_disease_id",
#         "norm_disease_name",
#     ]
#     df = input_df.loc[:, col_list]
#     df.columns = [string.lower() for string in col_list]

#     # Fill NaNs and return
#     return df.fillna("")


# def clean_gt_data(
#     pos_df: pd.DataFrame,
#     neg_df: pd.DataFrame,
#     endpoint: str,
#     conflate: bool,
#     drug_chemical_conflate: bool,
#     batch_size: int,
#     parallelism: int,
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     """Clean ground truth data.

#     Args:
#         pos_df: positive ground truth data.
#         neg_df: negative ground truth data.
#         endpoint: endpoint of the synonymizer.
#         conflate: whether to conflate
#         drug_chemical_conflate: whether to conflate drug and chemical
#         batch_size: batch size
#         parallelism: parallelism
#     Returns:
#         Cleaned ground truth data.
#     """
#     # Synonymize source and target IDs for both positive and negative ground truth data
#     for df in [pos_df, neg_df]:
#         for col in ["source", "target"]:
#             json_parser = parse("$.id.identifier")
#             node_id_map = batch_map_ids(
#                 frozenset(df[col]),
#                 api_endpoint=endpoint,
#                 batch_size=batch_size,
#                 parallelism=parallelism,
#                 conflate=conflate,
#                 drug_chemical_conflate=drug_chemical_conflate,
#                 json_parser=json_parser,
#             )
#             df[col] = df[col].map(node_id_map)

#     return pos_df.dropna(subset=["source", "target"]).drop_duplicates(), neg_df.dropna(
#         subset=["source", "target"]
#     ).drop_duplicates()


# # FUTURE: Remove the functions once we have tags embedded in the disease list
# @inject_object()
# def generate_tag(
#     disease_list: List, model: Dict, definitions: str = None, synonyms: str = None, raw_prompt: str = None
# ) -> List:
#     """Temporary function to generate tags based on provided prompts and params through OpenAI API call.

#     This function is temporary and will be removed once we have tags embedded in the disease list.

#     Args:
#         disease_list: list- list of disease for which tags should be generated.
#         definitions: str - (optional) definition of the disease, needed for prompts requiring multiple inputs.
#         synonyms: str - (optional) synonyms of the disease, needed for prompts requiring multiple inputs.
#         prompt: str - prompt for the tag generation
#         llm_model: str - name of the llm model to use for tag generation
#     Returns
#         List of tags generated by the API call.
#     """
#     # Initialize the output parser
#     output_parser = CommaSeparatedListOutputParser()

#     # Generate tags
#     tag_list = []
#     for i, disease in enumerate(disease_list):
#         if (definitions is None) | (synonyms is None):
#             prompt = ChatPromptTemplate.from_messages(
#                 [SystemMessage(content=raw_prompt), HumanMessage(content=disease)]
#             )
#             formatted_prompt = prompt.format_messages(disease=disease)
#         else:
#             prompt = ChatPromptTemplate.from_messages(
#                 [
#                     SystemMessage(
#                         content=raw_prompt.format(disease=disease, synonym=synonyms[i], definition=definitions[i])
#                     )
#                 ]
#             )

#             formatted_prompt = prompt.format_messages(disease=disease, synonym=synonyms[i], definition=definitions[i])
#         response = model.invoke(formatted_prompt)
#         tags = output_parser.parse(response.content)
#         tag_list.append(", ".join(tags))
#     return tag_list


# def enrich_disease_list(disease_list: List, params: Dict) -> pd.DataFrame:
#     """Temporary function to enrich existing disease list with llm-generated tags.

#     This function  will be removed once we have tags embedded in the disease list.

#     Args:
#         disease_list: pd.DataFrame - merged disease_list with disease names column that will be used for tag generation
#         params: Dict - parameters dictionary specifying tag names, column names, and model params
#         llm_model:  - name of the llm model to use for tag generation
#     Returns
#         pd.DataFrame with x new tag columns (where x corresponds to number of tags specified in params)
#     """
#     disease_list = disease_list
#     for input_type in ["single_input", "multiple_input"]:
#         input_params = params[input_type]
#         for tag, tag_params in input_params.items():
#             # Check if tag is already in disease list
#             if tag in disease_list.columns:
#                 continue

#             print(f"Applying tag: '{tag}' to disease list")

#             input_col = tag_params["input_params"]["input_col"]
#             output_col = tag_params["input_params"]["output_col"]
#             raw_prompt = tag_params["input_params"]["prompt"]
#             model = tag_params["model_params"]

#             # Check whether the tag needs a single or multiple inputs
#             if input_type == "single_input":
#                 disease_list[output_col] = generate_tag(
#                     disease_list=disease_list[input_col], raw_prompt=raw_prompt, model=model
#                 )
#             else:
#                 definition_col = tag_params["input_params"]["definition"]
#                 synonym_col = tag_params["input_params"]["synonyms"]
#                 disease_list[output_col] = generate_tag(
#                     disease_list=disease_list[input_col],
#                     definitions=disease_list[definition_col],
#                     synonyms=disease_list[synonym_col],
#                     raw_prompt=raw_prompt,
#                     model=model,
#                 )
#     return disease_list
