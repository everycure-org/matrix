from typing import Dict, Tuple

# from matrix.pipelines.integration.nodes import batch_map_ids
# from jsonpath_ng import parse
import pandas as pd
from matplotlib.figure import Figure
from . import visualisers

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


def resolve_input_sheet(
    input_sheet: pd.DataFrame, drug_sheet: pd.DataFrame, disease_sheet: pd.DataFrame
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """Run inference on disease or list of diseases and drug list, and write to google sheets.

    Args:
        input_sheet: dataframe with users' input IDs.
        drug_sheet: drug list with normalized curies.
        disease_sheet: disease list with normalized curies.

    Returns:
        str: The inferred type ("inferPerDrug", "inferPerDisease", or "inferPerPair").

    Raises:
        ValueError: If both drug_id and disease_id are empty.
    """
    # Load drug/disases IDs and timestamp from the msot re
    drug_id = input_sheet["norm_drug_id"].values[-1]
    drug_name = input_sheet["norm_drug_name"].values[-1]
    disease_id = input_sheet["norm_disease_id"].values[-1]
    disease_name = input_sheet["norm_disease_name"].values[-1]
    timestamp = input_sheet["timestamp"].values[-1].split(" ")[0]
    # Rule-based check of the request type
    if isinstance(disease_id, float) and isinstance(drug_id, float):
        raise ValueError("Need to specify drug, disease, or both")
    elif isinstance(disease_id, float):
        drug_list = pd.DataFrame({"curie": [drug_id], "name": [drug_name]})
        disease_list = disease_sheet
        request = "Drug-centric predictions"
    elif isinstance(drug_id, float):
        drug_list = drug_sheet
        disease_list = pd.DataFrame({"curie": [disease_id], "name": [disease_name]})
        request = "Disease-centric predictions"
    else:
        drug_list = pd.DataFrame({"curie": [drug_id], "name": [drug_name]})
        disease_list = pd.DataFrame({"curie": [disease_id], "name": [disease_name]})
        request = "Drug-disease specific predictions"
    return {"time": timestamp, "request": request}, drug_list, disease_list


def visualise_treat_scores(scores: pd.DataFrame, infer_type: Dict, col_name: str) -> Figure:
    """Create visualisations based on the treat scores and store them in GCS/MLFlow.

    Args:
        scores: treat scores generated during the inference.
        infer_type: type of inference requested.
        col_name: name of the column with treat scores

    Returns:
        figure: figure saved locally and in MLFlow
    """
    # FUTURE: add more visualisations
    kde_plot = visualisers.create_kdeplot(scores, infer_type, col_name)
    return kde_plot
