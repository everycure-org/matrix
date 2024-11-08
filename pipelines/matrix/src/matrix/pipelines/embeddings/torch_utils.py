"""Utils needed for pytorch geometric integration."""

from typing import List, Dict
from graphdatascience import GraphDataScience
import torch
from tqdm import tqdm
import torch.nn.functional as F


def generate_dummy_model() -> torch.nn.Sequential:
    """
    Generate a dummy PyTorch model (required for GDS models).

    Returns:
        torch.nn.Sequential: A sequential model with a single linear layer.
    """
    return torch.nn.Sequential(
        torch.nn.Linear(1, 1),
    )


class ContrastiveLoss:
    def __init__(self, num_negative_samples: int, neg_sample_weights: float):
        """
        Initialize the ContrastiveLoss class.

        Args:
            num_negative_samples (int): The number of negative samples to generate.
            neg_sample_weights (float): The weight of the negative samples.
        """
        self.num_negative_samples = num_negative_samples
        self.neg_sample_weights = neg_sample_weights

    def compute(self, out: torch.Tensor, edge_label: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        """
        Compute the contrastive loss for a GNN model with internal negative sampling.

        Args:
            out (torch.Tensor): The output tensor from the model.
            edge_label (torch.Tensor): The ground truth labels for the edges.
            edge_label_index (torch.Tensor): The indices of the edges in the output tensor.

        Returns:
            torch.Tensor: The computed loss value.
        """
        # Calculate positive affinities via dot product
        pos_src = out[edge_label_index[0]]
        pos_dst = out[edge_label_index[1]]
        aff = (pos_src * pos_dst).sum(dim=-1)

        # Generate negative samples and calculate negative affinities via dot product
        neg_indices = torch.randint(
            0, out.size(0), (self.num_negative_samples, edge_label_index.size(1)), device=out.device
        )
        neg_samples = out[neg_indices]
        neg_aff = (pos_src.unsqueeze(1) * neg_samples).sum(dim=-1)

        # Calculate true and negative cross-entropy losses
        true_xent = F.binary_cross_entropy_with_logits(aff, torch.ones_like(aff), reduction="sum")
        negative_xent = F.binary_cross_entropy_with_logits(neg_aff, torch.zeros_like(neg_aff), reduction="sum")

        loss = true_xent + self.neg_sample_weights * negative_xent
        return loss


# def xent_loss(
#     out: torch.Tensor,
#     edge_label: torch.Tensor,
#     edge_label_index: torch.Tensor,
#     num_negative_samples: int,
#     neg_sample_weights: float,
# ) -> torch.Tensor:
#     """
#     Compute the contrastive loss for a GNN model with internal negative sampling.

#     Args:
#         out (torch.Tensor): The output tensor from the model.
#         edge_label (torch.Tensor): The ground truth labels for the edges.
#         edge_label_index (torch.Tensor): The indices of the edges in the output tensor.
#         num_neg_samples (int): The number of negative samples to generate.
#         neg_sample_weights (float): The weight of the negative samples.

#     Returns:
#         torch.Tensor: The computed loss value.
#     """
#     # Calculate positive affinities via dot product
#     pos_src = out[edge_label_index[0]]
#     pos_dst = out[edge_label_index[1]]
#     aff = (pos_src * pos_dst).sum(dim=-1)

#     # Generate negative samples and calculate negative affinities via dot product
#     neg_indices = torch.randint(0, out.size(0), (num_neg_samples, edge_label_index.size(1)), device=out.device)
#     neg_samples = out[neg_indices]
#     neg_aff = (pos_src.unsqueeze(1) * neg_samples).sum(dim=-1)

#     # Calculate true and negative cross-entropy losses
#     true_xent = F.binary_cross_entropy_with_logits(aff, torch.ones_like(aff), reduction="sum")
#     negative_xent = F.binary_cross_entropy_with_logits(neg_aff, torch.zeros_like(neg_aff), reduction="sum")

#     loss = true_xent + neg_sample_weights * negative_xent
#     return loss


def prepare_graph_data(
    gds: GraphDataScience,
    graph_name: str,
    properties: bool = False,
    node_properties: str = None,
    edges_included: str = None,
) -> tuple[torch.Tensor, Dict[str, int], torch.Tensor]:
    """
    Prepare graph data for PyTorch Geometric models using Cypher queries.

    Args:
        gds (GraphDataScience): The Graph Data Science driver.
        graph_name (str): The name of the graph projection in Neo4j.
        properties (bool, optional): Whether to return node properties. Defaults to False.
        node_properties (str, optional): The properties to include in the node query. Defaults to None.
        edges_included (str, optional): The edges to include in the edge query. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - edge_index (torch.Tensor): The edge index tensor for PyG.
            - node_to_index (dict): A mapping from Neo4j node IDs to consecutive indices.
            - node_features (torch.Tensor): The node features tensor.
    """
    # Get all nodes depending on the properties flag
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
    if node_result.empty:
        raise ValueError(f"No nodes found in graph {graph_name}")

    node_to_index = {}
    node_features = [] if properties else None
    for idx, row in node_result.iterrows():
        node_to_index[row["nodeId"]] = idx
        if properties:
            node_features.append(row["propertyValue"])

    # Get all edges
    edge_result = gds.run_cypher(f"""
        CALL gds.graph.relationshipProperties.stream('{graph_name}',["{edges_included}"])
        YIELD sourceNodeId, targetNodeId, propertyValue
        WHERE propertyValue = 1
        RETURN sourceNodeId, targetNodeId
    """)
    if edge_result.empty:
        raise ValueError(f"No edges found in graph {graph_name}")

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


def train_n2v_model(model, dataloader, epochs, optimizer, device: str = "cpu") -> List[float]:
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


def train_gnn_model(
    model, dataloader, epochs, optimizer, device: str = "cpu", criterion: torch.nn.Module = None
) -> List[float]:
    """
    Trains a model on a given dataloader for a specified number of epochs.

    Args:
        model: The model to be trained.
        dataloader: The dataloader for the training data.
        epochs (int): The number of epochs to train the model.
        optimizer: The optimizer to use for training.
        device (str, optional): The device to use for training. Defaults to 'cpu'.
        criterion (torch.nn.Module, optional): The criterion to use for training. Defaults to None.

    Returns:
        List[float]: A list of total loss values for each epoch.
    """
    model.train()
    total_loss = []
    for epoch in tqdm(range(epochs), desc="Training"):
        epoch_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion.compute(out, batch.edge_label, batch.edge_label_index)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}; Loss", avg_loss)
        total_loss.append(avg_loss)
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
