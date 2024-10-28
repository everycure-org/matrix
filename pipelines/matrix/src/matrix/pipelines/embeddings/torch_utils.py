"""Utils needed for pytorch geometric integration."""

from typing import List, Dict
from graphdatascience import GraphDataScience
import torch
from tqdm import tqdm
import torch.nn.functional as F


def generate_dummy_model():
    return torch.nn.Sequential(
        torch.nn.Linear(1, 1),
    )


def BCE_contrastive_loss(out, edge_label, edge_label_index):
    out_src = out[edge_label_index[0]]
    out_dst = out[edge_label_index[1]]
    pred = (out_src * out_dst).sum(dim=-1)
    return F.binary_cross_entropy_with_logits(pred, edge_label)


def prepare_graph_data(
    gds: GraphDataScience,
    graph_name: str,
    properties: bool = False,
    node_properties: str = None,
    edges_excluded: str = None,
):
    """
    Prepare graph data for PyTorch Geometric models using Cypher queries.

    Args:
        gds (GraphDataScience): The Graph Data Science driver.
        graph_name (str): The name of the graph projection in Neo4j.
        properties (bool, optional): Whether to return node properties. Defaults to False.
        params (dict, optional): Additional parameters for the Cypher queries. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - edge_index (torch.Tensor): The edge index tensor for PyG.
            - node_to_index (dict): A mapping from Neo4j node IDs to consecutive indices.
    """
    # Get all nodes
    if properties:
        node_query = f"""
            CALL gds.graph.streamNodeProperties('{graph_name}', ['{node_properties}'])
            YIELD nodeId, propertyValue
            RETURN nodeId, propertyValue
        """
    else:
        node_query = f"""
            CALL gds.graph.streamNodeProperties('{graph_name}', ['{node_properties}'])
            YIELD nodeId
            RETURN nodeId
        """
    node_result = gds.run_cypher(node_query)
    # Create node_to_index mapping
    node_to_index = {}
    node_features = [] if properties else None
    for idx, row in node_result.iterrows():
        node_to_index[row["nodeId"]] = idx
        if properties:
            node_features.append(row["propertyValue"])

    # Get all edges
    edge_result = gds.run_cypher(f"""
        CALL gds.graph.relationshipProperties.stream('{graph_name}',["{edges_excluded}"])
        YIELD sourceNodeId, targetNodeId, propertyValue
        WHERE propertyValue = 1
        RETURN sourceNodeId, targetNodeId
    """)

    # Convert edges to edge_index format
    edge_index = torch.tensor(
        [
            [node_to_index[row["sourceNodeId"]] for _, row in edge_result.iterrows()],
            [node_to_index[row["targetNodeId"]] for _, row in edge_result.iterrows()],
        ],
        dtype=torch.long,
    )

    if properties:
        node_features = torch.tensor(node_features, dtype=torch.float)
        return edge_index, node_to_index, node_features
    else:
        return edge_index, node_to_index


def train_model(model, dataloader, epochs, optimizer, device: str = "cpu") -> List[float]:
    """
    Trains a model on a given dataloader for a specified number of epochs.

    Args:
        model: The model to be trained.
        dataloader: The dataloader for the training data.
        epochs (int): The number of epochs to train the model.
        optimizer: The optimizer to use for training.
        device (str, optional): The device to use for training. Defaults to 'cpu'.

    Returns:
        List[float]: A list of total loss values for each epoch.
    """
    total_loss = []
    for _ in tqdm(range(epochs)):
        for pos_rw, neg_rw in dataloader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
    return total_loss


def write_embeddings(gds: GraphDataScience, embeddings: torch.Tensor, write_property: str, node_index: Dict[str, int]):
    """
    Write embeddings to the graph.

    Args:
        gds (GraphDataScience): The Graph Data Science object.
        embeddings (torch.Tensor): The embeddings tensor.
        write_property (str): The property name to write the embeddings to.
        node_to_index (Dict[str, int]): A dictionary mapping node IDs to their indices.
    """
    total_nodes = len(node_index)
    batch_size = 1000

    if isinstance(node_index, dict):
        node_ids = list(node_index.keys())
    else:
        node_ids = node_index

    if len(node_ids) != total_nodes:
        raise ValueError(f"Number of node IDs ({len(node_ids)}) does not match number of embeddings ({total_nodes})")

    for i in tqdm(range(0, total_nodes, batch_size)):
        batch_nodes = node_ids[i : i + batch_size]
        batch_embeddings = embeddings[i : i + batch_size].tolist()

        query = f"""
        UNWIND $batch AS row
        MATCH (n) WHERE id(n) = row.node_id
        SET n.{write_property} = row.embedding
        """

        batch_data = [
            {"node_id": int(node_id), "embedding": emb} for node_id, emb in zip(batch_nodes, batch_embeddings)
        ]
        gds.run_cypher(query, params={"batch": batch_data})
