"""Utils needed for pytorch geometric integration."""

from typing import List, Dict
from graphdatascience import GraphDataScience
import torch
from tqdm import tqdm


def generate_dummy_model():
    return torch.nn.Sequential(
        torch.nn.Linear(1, 1),
    )


def prepare_graph_data(gds: GraphDataScience, graph_name: str):
    """
    Prepare graph data for PyTorch Geometric models using Cypher queries.

    Args:
        gds (GraphDataScience): The Graph Data Science driver.
        graph_name (str): The name of the graph projection in Neo4j.

    Returns:
        tuple: A tuple containing:
            - edge_index (torch.Tensor): The edge index tensor for PyG.
            - node_to_index (dict): A mapping from Neo4j node IDs to consecutive indices.
    """
    # Get all nodes
    node_result = gds.run_cypher(f"""
        CALL gds.graph.streamNodeProperties('{graph_name}', ['pca_embedding'])
        YIELD nodeId
        RETURN nodeId
    """)

    # Create node_to_index mapping
    node_to_index = {row["nodeId"]: idx for idx, row in node_result.iterrows()}

    # Get all edges
    edge_result = gds.run_cypher(f"""
        CALL gds.graph.relationships.stream('{graph_name}')
        YIELD sourceNodeId, targetNodeId
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

    return edge_index, node_to_index


# def prepare_graph_data(gds: GraphDataScience, graph_name: str):
#     """
#     Prepare graph data for PyTorch Geometric models.

#     Args:
#         gds (GraphDataScience): The Graph Data Science driver.
#         graph_name (str): The name of the graph projection in Neo4j.

#     Returns:
#         tuple: A tuple containing:
#             - edge_index (torch.Tensor): The edge index tensor for PyG.
#             - node_to_index (dict): A mapping from Neo4j node IDs to consecutive indices.
#     """
#     # Get all nodes and their Neo4j IDs
#     print(graph_name)
#     result = gds.run_cypher(f"""
#         CALL gds.graph.streamNodeProperties('{graph_name}', ['pca_embedding'])
#         YIELD nodeId
#         RETURN nodeId
#     """)
#     print("Node query result structure:")
#     print(result)
#     print(type(result))
#     print(result.columns)

#     # Create node_to_index mapping
#     node_to_index = {}
#     for idx, row in result.iterrows():
#         node_to_index[row['nodeId']] = idx

#     # Get all edges, considering the 'include_in_graphsage' property
#     edge_result = gds.run_cypher(f"""
#         CALL gds.graph.relationships.stream('{graph_name}')
#         YIELD sourceNodeId, targetNodeId, relationshipProperties
#         WHERE relationshipProperties.include_in_graphsage = 1
#         RETURN sourceNodeId, targetNodeId
#     """)
#     print("Edge query result structure:")
#     print(edge_result)
#     print(type(edge_result))
#     print(edge_result.columns)

#     # Create edges list
#     edges = []
#     for _, row in edge_result.iterrows():
#         source = node_to_index[row['sourceNodeId']]
#         target = node_to_index[row['targetNodeId']]
#         edges.append((source, target))

#     # Create edge_index tensor
#     edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

#     return edge_index, node_to_index

# def create_node_index(gds: GraphDataScience, graph_name: str, node_label: str = 'Entity', node_property: str = 'pca_embedding') -> Dict[int, int]:
#     """
#     Create a mapping of node IDs to their indices for nodes with the specified label.

#     Args:
#         gds (GraphDataScience): The Graph Data Science object.
#         graph_name (str): The name of the graph.
#         node_label (str, optional): The label of the nodes to index. Defaults to 'Entity'.
#         node_property (str, optional): The node property to use as the identifier. Defaults to 'pca_embedding'.

#     Returns:
#         Dict[int, int]: A dictionary mapping node IDs to their indices.
#     """
#     query = f"""
#     CALL gds.graph.nodeProperties.stream('{graph_name}', ['{node_property}'], ['{node_label}'])
#     YIELD nodeId, propertyValue
#     RETURN nodeId, propertyValue
#     """
#     result = gds.run_cypher(query)
#     return {row['propertyValue']: row['nodeId'] for row in result}

# def generate_edge_index(gds: GraphDataScience, name: str) -> torch.Tensor:
#     """
#     Generates an edge index ten sor from a graph in Graph Data Science.

#     Args:
#         gds (GraphDataScience): The Graph Data Science object.
#         name (str): The name of the graph.

#     Returns:
#         torch.Tensor: A tensor representing the edge index, with shape (2, num_edges) and dtype torch.long.
#     """
#     edges = gds.run_cypher(
#             f"CALL gds.graph.relationships.stream('{name}') YIELD sourceNodeId, targetNodeId RETURN sourceNodeId, targetNodeId"
#         )
#     return torch.tensor(edges[["sourceNodeId", "targetNodeId"]].values.T, dtype=torch.long)


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
    node_ids = list(node_index.values())
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
