"""GDS Algorithms Classes."""

from typing import List, Any, Optional, Dict
from graphdatascience import GraphDataScience
import torch
from torch_geometric.nn import Node2Vec, GraphSAGE
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.data import Data
from . import torch_utils as tu
from .gds_algorithms import GraphAlgorithm


class PygNode2Vec(GraphAlgorithm):
    """PyTorch Geometric Node2Vec algorithm class."""

    def __init__(
        self,
        walk_length: int = 80,
        walks_per_node: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        num_negative_samples: int = 1,
        embedding_dim: int = 512,
        random_seed: Optional[int] = None,
        concurrency: int = 4,
        epochs: int = 10,
        sparse: bool = True,  # False,
        context_size: int = 10,
        batch_size: int = 128,
        num_workers: int = 0,
        learning_rate: float = 0.01,
        optimizer: str = "SparseAdam",
    ):
        """PyTorch Geometric Node2Vec Attributes.

        For more information on the parameters, see https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.Node2Vec.html.
        """
        super().__init__(embedding_dim, random_seed, concurrency)
        self._walk_length = walk_length
        self._walks_per_node = walks_per_node
        self._p = p
        self._q = q
        self._num_negative_samples = num_negative_samples
        self._model = None
        self._loss_lst = None
        self._epochs = epochs
        self._batch_size = batch_size
        self._context_size = context_size
        self._sparse = sparse
        self._num_workers = num_workers
        self._lr = learning_rate
        self._optimizer = optimizer

    def run(
        self,
        gds: GraphDataScience,
        graph: Any,
        model_name: str,
        write_property: str,
        subgraph: str,
        device: str = "cpu",
        node_projection: dict = None,
        config: dict = None,
    ):
        """Train the algorithm.

        Args:
            gds: GraphDataScience instance, used to project graph and get GDS models
            graph: Graph object that will be used to train the model (dummy; used by GDS)
            model_name: Name of the graph model to be projected in Neo4j (dummy; used by GDS)
            write_property: Property/Attribute name to write embeddings to (dummy)
            subgraph: Subgraph name to use for training (e.g. if GDS filtering is applied, we want subgraph instead of whole graph)
            device: Device to use for training
            node_projection: Node projection dictionary, specifies which node properties to use for training
            config: Configuration dictionary
        """
        edge_index, _ = tu.prepare_graph_data(
            gds,
            subgraph,
            properties=False,
            node_properties=node_projection["Entity"]["properties"],
            edges_included=config["relationshipProperties"]["include_in_graphsage"]["property"],
        )
        # Initialize the Node2Vec model
        self._model = Node2Vec(
            edge_index,
            embedding_dim=self._embedding_dim,
            walk_length=self._walk_length,
            context_size=self._context_size,
            walks_per_node=self._walks_per_node,
            p=self._p,
            q=self._q,
            num_negative_samples=self._num_negative_samples,
            sparse=self._sparse,
        ).to(device)

        # Initialize optimizer
        optimizer_class = getattr(torch.optim, self._optimizer)
        optimizer = optimizer_class(list(self._model.parameters()), lr=self._lr)

        # Create a loader and train the model
        loader = self._model.loader(batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)

        # Set Neural Network in training mode and train the model
        self._model.train()
        self._loss_lst = tu.train_n2v_model(self._model, loader, self._epochs, optimizer, device)

        return self._model, {"loss": self._loss_lst}

    def return_loss(self):
        """Return loss."""
        return self._loss_lst

    def predict_write(
        self,
        gds: GraphDataScience,
        graph: Any,
        model_name: str,
        state_dict: torch.Tensor,
        write_property: str,
        graph_name: str,
        device: str = "cpu",
        node_projection: dict = None,
        config: dict = None,
    ):
        """Predict and save.

        Args:
            gds: GraphDataScience instance, used to project graph and get GDS models
            graph: Graph object that will be used to train the model (dummy; used by GDS)
            model_name: Name of the graph model to be projected in Neo4j (dummy; used by GDS)
            state_dict: State dictionary of the model to be loaded for prediction
            write_property: Property/Attribute name to write embeddings to
            graph_name: Subgraph name to use for inference (e.g. if GDS filtering is applied, we want subgraph instead of whole graph)
            device: Device to use for training
            node_projection: Node projection dictionary, specifies which node properties to use for training
            config: Configuration dictionary
        """
        # Generate edge index and node index for torch
        edge_index, node_index = tu.prepare_graph_data(
            gds,
            graph_name,
            properties=False,
            node_properties=node_projection["Entity"]["properties"],
            edges_included=config["relationshipProperties"]["include_in_graphsage"]["property"],
        )

        # Generate embeddings
        n2vec = Node2Vec(
            edge_index,
            embedding_dim=self._embedding_dim,
            walk_length=self._walk_length,
            context_size=self._walks_per_node,
            walks_per_node=self._walks_per_node,
            p=self._p,
            q=self._q,
            num_negative_samples=self._num_negative_samples,
            sparse=self._sparse,
        ).to(device)

        # Overwrite state dict with actual model
        n2vec.load_state_dict(state_dict)
        n2vec.eval()

        with torch.no_grad():
            embeddings = n2vec.embedding.weight.cpu()
        tu.write_embeddings(gds, embeddings, write_property, node_index)


class PygGraphSage(GraphAlgorithm):
    """PyTorch Geometric GraphSAGE algorithm class."""

    def __init__(
        self,
        num_layers: int = 2,
        hidden_channels: int = 256,
        embedding_dim: int = 512,
        random_seed: Optional[int] = None,
        concurrency: int = 4,
        epochs: int = 10,
        batch_size: int = 128,
        num_neighbors: List[int] = [25, 10],
        num_workers: int = 0,
        learning_rate: float = 0.01,
        optimizer: str = "Adam",
        aggregator: str = "mean",
        dropout: float = 0.0,
        neg_sampling_ratio: float = 1.0,
        criterion: str = None,
    ):
        """PyTorch Geometric GraphSAGE Attributes.


        For more information on the parameters, see https://pytorch-geometric.readthedocs.io/en/2.4.0/generated/torch_geometric.nn.models.GraphSAGE.html.
        """
        super().__init__(embedding_dim, random_seed, concurrency)
        self._num_layers = num_layers
        self._hidden_channels = hidden_channels
        self._model = None
        self._loss_lst = None
        self._criterion = criterion
        self._epochs = epochs
        self._batch_size = batch_size
        self._num_neighbors = num_neighbors
        self._num_workers = num_workers
        self._lr = learning_rate
        self._optimizer = optimizer
        self._aggregator = aggregator
        self._dropout = dropout
        self._neg_sampling_ratio = neg_sampling_ratio

    def run(
        self,
        gds: GraphDataScience,
        graph: Any,
        model_name: str,
        write_property: str,
        subgraph: str,
        device: str = "cpu",
        node_projection: dict = None,
        config: dict = None,
    ) -> tuple[torch.nn.Sequential, Dict[str, Any]]:
        """Train the algorithm.

        Args:
            gds: GraphDataScience instance, used to project graph and get GDS models
            graph: Graph object that will be used to train the model (dummy; used by GDS)
            model_name: Name of the graph model to be projected in Neo4j (dummy; used by GDS)
            write_property: Property/Attribute name to write embeddings to (dummy)
            subgraph: Subgraph name to use for training (e.g. if GDS filtering is applied, we want subgraph instead of whole graph)
            device: Device to use for training
            node_projection: Node projection dictionary, specifies which node properties to use for training
            config: Configuration dictionary
        """
        # Convert the graph to PyTorch Geometric format
        edge_index, _, x = tu.prepare_graph_data(
            gds,
            subgraph,
            properties=True,
            node_properties=node_projection["Entity"]["properties"],
            edges_included=config["relationshipProperties"]["include_in_graphsage"]["property"],
        )
        data = Data(x=x, edge_index=edge_index)

        # Initialize the GraphSAGE model
        self._model = GraphSAGE(
            in_channels=x.shape[1],
            hidden_channels=self._hidden_channels,
            num_layers=self._num_layers,
            out_channels=self._embedding_dim,
            dropout=self._dropout,
            aggr=self._aggregator,
        ).to(device)

        # Initialize optimizer
        optimizer_class = getattr(torch.optim, self._optimizer)
        optimizer = optimizer_class(self._model.parameters(), lr=self._lr)

        # Create data loader
        loader = LinkNeighborLoader(
            data=data,
            num_neighbors=self._num_neighbors,
            batch_size=self._batch_size,
            shuffle=True,
            neg_sampling_ratio=self._neg_sampling_ratio,
            num_workers=self._num_workers,
        )
        # Train the model
        self._loss_lst = tu.train_gnn_model(
            model=self._model,
            dataloader=loader,
            epochs=self._epochs,
            optimizer=optimizer,
            device="cpu",
            criterion=self._criterion,
        )
        return self._model, {"loss": self._loss_lst}

    def return_loss(self) -> List[float]:
        """Return loss."""
        return self._loss_lst

    def predict_write(
        self,
        gds: GraphDataScience,
        graph: Any,
        model_name: str,
        state_dict: torch.Tensor,
        write_property: str,
        graph_name: str,
        device: str = "cpu",
        node_projection: dict = None,
        config: dict = None,
    ):
        """Predict and save.
        Args:
            gds: GraphDataScience instance, used to project graph and get GDS models
            graph: Graph object that will be used to train the model (dummy; used by GDS)
            model_name: Name of the graph model to be projected in Neo4j (dummy; used by GDS)
            state_dict: State dictionary of the model to be loaded for prediction
            write_property: Property/Attribute name to write embeddings to
            graph_name: Subgraph name to use for inference (e.g. if GDS filtering is applied, we want subgraph instead of whole graph)
            device: Device to use for training
            node_projection: Node projection dictionary, specifies which node properties to use for training
            config: Configuration dictionary
        """
        # Get edge index and node features
        edge_index, node_to_index, x = tu.prepare_graph_data(
            gds,
            graph_name,
            properties=True,
            node_properties=node_projection["Entity"]["properties"],
            edges_included=config["relationshipProperties"]["include_in_graphsage"]["property"],
        )

        # Initialize the GraphSAGE model
        model = GraphSAGE(
            in_channels=x.shape[1],
            hidden_channels=self._hidden_channels,
            num_layers=self._num_layers,
            out_channels=self._embedding_dim,
            dropout=self._dropout,
            aggr=self._aggregator,
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        # Generate embeddings
        with torch.no_grad():
            embeddings = model(x.to(device), edge_index.to(device)).cpu()

        tu.write_embeddings(gds, embeddings, write_property, list(node_to_index.keys()))
