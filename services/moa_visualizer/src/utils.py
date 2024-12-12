import os
import pandas as pd
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
import streamlit as st
from config import settings, ont_urls, display_cols

# TODO
# - Clean up table formatting
# - Add some hyperlinks to entities (e.g. link to UniProt)
# - Probably some interactive elements to the graphs
# - Make a class to handle data build


def get_pair_info_from_db(moa_db_path: str = settings.moa_db_path, path_number: str = "all") -> pd.DataFrame:
    """
    Reads the MOA database from a SQLite database file

    Args:
        moa_db_path (str): Path to the SQLite database file
        path_number (str): The number of the path to return. Defaults to 'all',
        or 'two_hop' or 'three_hop'

    Returns:
        pd.DataFrame: DataFrame containing the MOA database contents
    """
    if not os.path.exists(moa_db_path):
        raise FileNotFoundError(f"Database file not found at {moa_db_path}")
    if path_number not in ["all", "two_hop", "three_hop"]:
        raise ValueError("path_number must be 'all', 'two_hop' or 'three_hop'")

    query = f"""SELECT pair_id, "Drug Name", "Disease Name" FROM 
    {path_number}_pair_info_all"""

    df = pd.read_sql_query(query, f"sqlite:///{moa_db_path}")
    df = df.rename(columns={"Drug Name": "drug_name", "Disease Name": "disease_name"})
    return df


@st.cache_data
def list_available_pairs_df(input_path: str = settings.moa_db_path, path_number: str = "two-hop") -> pd.DataFrame:
    df = get_pair_info_from_db(moa_db_path=input_path, path_number=path_number)
    df["drug_name"] = df["drug_name"].str.capitalize()
    df["disease_name"] = df["disease_name"].str.capitalize()
    return df


def get_moa_predictions_from_db(input_path: str, pair_id: str, path_number: str) -> pd.DataFrame:
    if path_number not in ["two_hop", "three_hop"]:
        raise ValueError("path_number must be 'two_hop' or 'three_hop'")
    query = f"""SELECT * FROM {path_number}_predictions_all 
    WHERE pair_id = '{pair_id}' ORDER BY MOA_score DESC"""
    df = pd.read_sql_query(query, f"sqlite:///{input_path}")
    return df


def parse_hop_parameter(hop_parameter: bool) -> str:
    return "two_hop" if not hop_parameter else "three_hop"


def combine_moa_predictions_and_pair_info(moa_predictions: pd.DataFrame, pair_info: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(moa_predictions, pair_info, on="pair_id")


def select_last_edge(edges: str) -> str:
    if not edges:
        return ""
    last_edge = edges.split(",")[-1]

    if last_edge.strip().endswith("*"):
        last_edge = last_edge.strip()[:-1]

    return last_edge.strip()


def build_external_urls(id_string: str) -> str:
    # Maybe make a dict of ontologies and their URLs for constants
    for ont, url in ont_urls.model_fields.items():
        if ont in id_string:
            # Maybe a better way is to regex out the last bit of the string and keep that only
            if ont == "UniProtKB":
                return f"{url}{id_string.replace(ont + ':', '')}"
            if ont == "CHEMBL":
                return f"{url}{id_string.replace(ont + '.COMPOUND:', '')}"
            else:
                return f"{url}{id_string}"


def display_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Displays a table with cleaner column names by reordering and subsetting
    to only include columns that exist in DISPLAY_COLS
    """
    rename_cols = dict(zip(display_cols.all_keys, display_cols.all_columns))
    cols_to_keep = [col for col in display_cols.all_keys if col in df.columns]
    return df[cols_to_keep].rename(columns=rename_cols)


class FlowDiagram:
    def __init__(self):
        self.nodes = []
        self.edges = []

    @staticmethod
    def generate_edges(moa_prediction: pd.DataFrame) -> list[StreamlitFlowEdge]:
        """
        Generates edges for the graph based on the MOA predictions output.
        Assumes that the intermediate nodes are named intermediate_name_1,
        intermediate_name_2, etc. and that the predicates are named
        predicates_1, predicates_2, etc. Assumes that intermediate_1 will
        always be connected to the drug and final intermediate will always be
        connected to the disease.
        """
        edge_style = {"fontSize": "5px", "padding": 0, "width": "100px"}

        edges = []
        for _, row in moa_prediction.iterrows():
            intermediate_cols = [col for col in row.index if "intermediate_name_" in col]
            intermediate_cols.sort(key=lambda x: int(x.split("_")[-1]))

            # Create first edge from drug to first intermediate
            edges.append(
                StreamlitFlowEdge(
                    id=row["drug_name"] + "-" + row[intermediate_cols[0]],
                    source=row["drug_name"],
                    target=row[intermediate_cols[0]],
                    label=select_last_edge(row["predicates_1"]),
                    edge_type="default",
                    animated=False,
                    label_style=edge_style,
                )
            )

            # Create edges between intermediate nodes
            for i in range(len(intermediate_cols) - 1):
                curr_node = intermediate_cols[i]
                next_node = intermediate_cols[i + 1]
                predicate_num = i + 2

                edges.append(
                    StreamlitFlowEdge(
                        id=row[curr_node] + "-" + row[next_node],
                        source=row[curr_node],
                        target=row[next_node],
                        label=select_last_edge(row[f"predicates_{predicate_num}"]),
                        edge_type="default",
                        animated=False,
                        label_style=edge_style,
                    )
                )

            # Create final edge from last intermediate to disease
            edges.append(
                StreamlitFlowEdge(
                    id=row[intermediate_cols[-1]] + "-" + row["disease_name"],
                    source=row[intermediate_cols[-1]],
                    target=row["disease_name"],
                    label=select_last_edge(row[f"predicates_{len(intermediate_cols)+1}"]),
                    edge_type="default",
                    animated=False,
                    label_style=edge_style,
                )
            )

            # Finally create a fake edge for MOA score so it appears in line
            edges.append(
                StreamlitFlowEdge(
                    id=row["disease_name"] + "-" + "MOA_score",
                    source=row["disease_name"],
                    target="MOA_score",
                    label="moa_score",
                    edge_type="default",
                    animated=False,
                    hidden=True,
                )
            )

        return edges

    @staticmethod
    def generate_nodes(moa_prediction: pd.DataFrame) -> list[StreamlitFlowNode]:
        """
        Generates nodes for the graph based on the MOA predictions output.
        Assumes that the intermediate nodes are named intermediate_name_1,
        intermediate_name_2, etc. and that the predicates are named
        predicates_1, predicates_2, etc. Assumes that intermediate_1 will
        always be connected to the drug and final intermediate will always be
        connected to the disease.
        """
        node_style = {"fontSize": "5px", "padding": 0, "width": "100px"}
        pixel_spacing = 200
        intermediate_cols = [col for col in moa_prediction.columns if "intermediate_name_" in col]
        intermediate_cols.sort(key=lambda x: int(x.split("_")[-1]))
        num_intermediates = len(intermediate_cols)
        total_nodes = num_intermediates + 2  # Add drug and disease nodes

        for _, row in moa_prediction.iterrows():
            nodes = []

            # Add drug node first
            nodes.append(
                StreamlitFlowNode(
                    id=row["drug_name"],
                    pos=(100, 100),
                    data={"content": row["drug_name"].capitalize()},
                    animated=False,
                    node_type="input",
                    source_position="right",
                    style=dict(node_style, **{"backgroundColor": "#77DD77"}),
                    draggable=False,
                )
            )

            # Add intermediate nodes in order
            for i, col in enumerate(intermediate_cols, 1):
                nodes.append(
                    StreamlitFlowNode(
                        id=row[col],
                        pos=(100 + (i * pixel_spacing), 100),
                        data={"content": row[col]},
                        animated=False,
                        target_position="left",
                        source_position="right",
                        style=dict(node_style, **{"backgroundColor": "#FF964F"}),
                        draggable=False,
                    )
                )

            # Add disease node last
            nodes.append(
                StreamlitFlowNode(
                    id=row["disease_name"],
                    pos=(100 + (total_nodes - 1) * pixel_spacing, 100),
                    data={"content": row["disease_name"].capitalize()},
                    animated=False,
                    node_type="output",
                    target_position="left",
                    style=dict(node_style, **{"backgroundColor": "#B3EBF2"}),
                    draggable=False,
                )
            )

            nodes.append(
                StreamlitFlowNode(
                    id="MOA_score",
                    pos=(225 + (total_nodes - 1) * pixel_spacing, 100),
                    data={"content": """MoA score: {:.3f}""".format(row["MOA_score"])},
                    animated=False,
                    node_type="default",
                    style={"fontSize": "6px", "padding": 0, "width": "70px"},
                    draggable=False,
                )
            )

        return nodes

    def create_graph_state(self, moa_prediction_df: pd.DataFrame) -> StreamlitFlowState:
        nodes = self.generate_nodes(moa_prediction_df)
        edges = self.generate_edges(moa_prediction_df)
        return StreamlitFlowState(nodes, edges)

    @staticmethod
    def render_graph(
        graph_state: StreamlitFlowState, graph_label: str, interactivity: bool = False
    ) -> StreamlitFlowState:
        """
        Draws a graph based on MoAs for those that have predictions
        """
        return streamlit_flow(
            graph_label,
            graph_state,
            fit_view=True,
            show_minimap=False,
            show_controls=interactivity,
            pan_on_drag=interactivity,
            allow_zoom=interactivity,
            hide_watermark=True,
            height=100,
        )
