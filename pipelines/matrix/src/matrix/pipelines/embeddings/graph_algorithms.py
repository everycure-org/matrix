from abc import ABC, abstractmethod
from typing import List, Any, Optional
from graphdatascience import GraphDataScience
import torch
from torch_geometric.nn import Node2Vec, GraphSAGE
from torch_geometric.loader import LinkNeighborLoader
from tqdm import tqdm
from torch_geometric.data import Data
from . import torch_utils as tu


class GraphAlgorithm(ABC):
    """Base class for Graph Algorithms, stores all arguments which can be used by child classes."""

    def __init__(self, embedding_dim: int = 512, random_seed: int = None, concurrency: int = 4):
        """Common Attributes."""
        self._embedding_dim = embedding_dim
        self._random_seed = random_seed
        self._concurrency = concurrency

    @abstractmethod
    def run(self, gds: GraphDataScience, graph: Any, model_name: str, write_property: str):
        """Run training base constructor."""
        pass

    @abstractmethod
    def predict_write(self, gds: GraphDataScience, graph: Any, model_name: str, write_property: str):
        """Predict and write constructors."""
        pass

    @abstractmethod
    def return_loss(self):
        """Return loss constructor."""
        pass


class GDSGraphSage(GraphAlgorithm):
    """GraphSAGE algorithm class. For more information see https://neo4j.com/docs/graph-data-science/current/machine-learning/node-embeddings/graph-sage/."""

    def __init__(
        self,
        sample_sizes: List[int] = [25, 10],
        epochs: int = 10,
        learning_rate: float = 0.01,
        iterations: int = 10,
        tolerance: float = 1e-8,
        search_depth: int = 5,
        aggregator: str = "mean",
        batch_size: int = 256,
        negative_sampling_weight: int = 20,
        activation_function: str = "sigmoid",
        feature_properties: str = "*",
        embedding_dim: int = 512,
        random_seed: Optional[int] = None,
        penalty_l2: float = 0.0,
        concurrency: int = 4,
    ):
        """GraphSAGE attributes."""
        super().__init__(embedding_dim, random_seed, concurrency)
        self._sample_sizes = sample_sizes
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._iterations = iterations
        self._aggregator = aggregator
        self._tolerance = tolerance
        self._search_depth = search_depth
        self._batch_size = batch_size
        self._negative_sampling_weight = negative_sampling_weight
        self._activation_function = activation_function
        self._feature_properties = feature_properties
        self._penalty_l2 = penalty_l2
        self._loss = None

    def run(self, gds: GraphDataScience, graph: Any, model_name: str, write_property: str):
        """Train the algorithm."""
        model, attr = gds.beta.graphSage.train(
            graph,
            modelName=model_name,
            concurrency=self._concurrency,
            sampleSizes=self._sample_sizes,
            learningRate=self._learning_rate,
            maxIterations=self._iterations,
            aggregator=self._aggregator,
            tolerance=self._tolerance,
            embeddingDimension=self._embedding_dim,
            batchSize=self._batch_size,
            randomSeed=self._random_seed,
            epochs=self._epochs,
            searchDepth=self._search_depth,
            negativeSampleWeight=self._negative_sampling_weight,
            activationFunction=self._activation_function,
            featureProperties=self._feature_properties,
            penaltyL2=self._penalty_l2,
        )
        self._loss = attr["modelInfo"]["metrics"]["iterationLossesPerEpoch"][0]
        model = tu.generate_dummy_model()
        return model, attr

    def return_loss(self):
        """Return loss."""
        return self._loss

    def predict_write(
        self,
        gds: GraphDataScience,
        graph: Any,
        model_name: str,
        write_property: str,
        graph_name: str,
        state_dict: torch.Tensor,
    ):
        """Predict and save."""
        model = gds.model.get(model_name)
        model.predict_write(graph, writeProperty=write_property)


class GDSNode2Vec(GraphAlgorithm):
    """Node2Vec algorithm class."""

    def __init__(
        self,
        walk_length: int = 80,
        walks_per_node: int = 10,
        in_out_factor: float = 1.0,
        return_factor: float = 1.0,
        iterations: int = 10,
        positive_sampling_factor: int = 0.001,
        relationship_weight_property: Optional[str] = None,
        negative_sampling_exponent: int = 0.75,
        negative_sampling_rate: int = 5,
        window_size: int = 10,
        initial_learning_rate: float = 0.01,
        min_learning_rate: float = 0.0001,
        embedding_dim: int = 512,
        random_seed: Optional[int] = None,
        concurrency: int = 4,
        walk_buffer_size: int = 1000,
    ):
        """Node2Vec Attributes. For more information see  https://neo4j.com/docs/graph-data-science/current/machine-learning/node-embeddings/node2vec/."""
        super().__init__(embedding_dim, random_seed, concurrency)
        self._walk_length = walk_length
        self._walks_per_node = walks_per_node
        self._in_out_factor = in_out_factor
        self._return_factor = return_factor
        self._relationship_weight_property = relationship_weight_property
        self._iterations = iterations
        self._positive_sampling_factor = positive_sampling_factor
        self._negative_sampling_exponent = negative_sampling_exponent
        self._negative_sampling_rate = negative_sampling_rate
        self._window_size = window_size
        self._initial_learning_rate = initial_learning_rate
        self._min_learning_rate = min_learning_rate
        self._walk_buffer_size = walk_buffer_size
        self._loss = None

    def run(self, gds: GraphDataScience, graph: Any, model_name: str, write_property: str):
        """Trains the algorithm and writes embeddings.

        Node2Vec has no separate training and inference steps (unlike GraphSage),
        so both are done via the same write function. This function also
        produces a loss attribute, needed for plotting convergence.
        """
        attr = gds.node2vec.write(
            G=graph,
            writeProperty=write_property,
            walkLength=self._walk_length,
            walksPerNode=self._walks_per_node,
            embeddingDimension=self._embedding_dim,
            inOutFactor=self._in_out_factor,
            returnFactor=self._return_factor,
            relationshipWeightProperty=self._relationship_weight_property,
            iterations=self._iterations,
            positiveSamplingFactor=self._positive_sampling_factor,
            negativeSamplingRate=self._negative_sampling_rate,
            negativeSamplingExponent=self._negative_sampling_exponent,
            windowSize=self._window_size,
            initialLearningRate=self._initial_learning_rate,
            minLearningRate=self._min_learning_rate,
            randomSeed=self._random_seed,
            walkBufferSize=self._walk_buffer_size,
        )
        self._loss = [int(x) for x in attr["lossPerIteration"]]

        # Dummy tensor dataset
        model = tu.generate_dummy_model()

        # Save the model using PyTorchDataset
        return model, attr

    def predict_write(
        self,
        gds: GraphDataScience,
        graph: Any,
        model_name: str,
        write_property: str,
        graph_name: str,
        state_dict: torch.Tensor,
    ):
        """Dummy function as Node2vec gets written in the 'run' step due to no separate predict_write function (unlike GraphSage)."""
        return

    def return_loss(self):
        """Return and save."""
        return self._loss


# Torch Geometric models


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
        """PyTorch Geometric Node2Vec Attributes."""
        super().__init__(embedding_dim, random_seed, concurrency)
        self._walk_length = walk_length
        self._walks_per_node = walks_per_node
        self._p = p
        self._q = q
        self._num_negative_samples = num_negative_samples
        self._model = None
        self._loss = None
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
        relationship_projection: dict = None,
        node_projection: dict = None,
        config: dict = None,
    ):
        """Train the algorithm."""
        node_properties = node_projection["Entity"]["properties"]
        edges_excluded = config["relationshipProperties"]["include_in_graphsage"]["property"]
        edge_index, _ = tu.prepare_graph_data(
            gds, subgraph, properties=False, node_properties=node_properties, edge_excluded=edges_excluded
        )
        # edge_index = tu.generate_edge_index(gds, subgraph)

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
        self._model.train()

        self._loss = tu.train_model(self._model, loader, self._epochs, optimizer, device)

        return self._model, {"loss": self._loss}

    def return_loss(self):
        """Return loss."""
        return self._loss

    def predict_write(
        self,
        gds: GraphDataScience,
        graph: Any,
        model_name: str,
        state_dict: torch.Tensor,
        write_property: str,
        graph_name: str,
        device: str = "cpu",
        projection: dict = None,
        relationship_projection: dict = None,
        node_projection: dict = None,
        config: dict = None,
    ):
        """Predict and save."""
        # get edge index
        node_properties = node_projection["Entity"]["properties"]
        edges_excluded = config["relationshipProperties"]["include_in_graphsage"]["property"]

        edge_index, node_index = tu.prepare_graph_data(
            gds, graph_name, properties=False, node_properties=node_properties, edges_excluded=edges_excluded
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
            sparse=True,
        ).to(device)

        # Overwrite state dict with actual model
        n2vec.load_state_dict(state_dict)
        n2vec.eval()

        with torch.no_grad():
            embeddings = n2vec.embedding.weight.cpu()
        tu.write_embeddings(gds, embeddings, write_property, node_index)


class PygGraphSAGE(GraphAlgorithm):
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
        """PyTorch Geometric GraphSAGE Attributes."""
        super().__init__(embedding_dim, random_seed, concurrency)
        self._num_layers = num_layers
        self._hidden_channels = hidden_channels
        self._model = None
        self._loss = None
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
        relationship_projection: dict = None,
        node_projection: dict = None,
        config: dict = None,
    ):
        """Train the algorithm."""
        # Convert the graph to PyTorch Geometric format
        node_properties = node_projection["Entity"]["properties"]
        edges_excluded = config["relationshipProperties"]["include_in_graphsage"]["property"]

        edge_index, _, x = tu.prepare_graph_data(
            gds, subgraph, properties=True, node_properties=node_properties, edges_excluded=edges_excluded
        )
        data = Data(x=x, edge_index=edge_index)

        # Initialize the GraphSAGE model
        self._model = GraphSAGE(
            in_channels=x.shape[1],
            hidden_channels=self._hidden_channels,
            num_layers=self._num_layers,
            out_channels=x.shape[1],  # self._embedding_dim,
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
        self._model.train()
        self._loss = []
        for epoch in tqdm(range(self._epochs), desc="Training"):
            epoch_loss = 0
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = self._model(batch.x, batch.edge_index)
                loss = self._criterion(out, batch.edge_label, batch.edge_label_index)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                print(f"Epoch {epoch}; Loss", loss)

            avg_loss = epoch_loss / len(loader)
            self._loss.append(avg_loss)

        return self._model, {"loss": self._loss}

    def return_loss(self):
        """Return loss."""
        return self._loss

    def predict_write(
        self,
        gds: GraphDataScience,
        graph: Any,
        model_name: str,
        state_dict: torch.Tensor,
        write_property: str,
        graph_name: str,
        device: str = "cpu",
        relationship_projection: dict = None,
        node_projection: dict = None,
        config: dict = None,
    ):
        """Predict and save."""
        # Get edge index and node features
        node_properties = node_projection["Entity"]["properties"]
        edges_excluded = config["relationshipProperties"]["include_in_graphsage"]["property"]
        edge_index, node_to_index, x = tu.prepare_graph_data(
            gds, graph_name, properties=True, node_properties=node_properties, edges_excluded=edges_excluded
        )

        # Initialize the GraphSAGE model
        model = GraphSAGE(
            in_channels=x.shape[1],
            hidden_channels=self._hidden_channels,
            num_layers=self._num_layers,
            out_channels=x.shape[1],  # self._embedding_dim,
            dropout=self._dropout,
            aggr=self._aggregator,
        ).to(device)
        # Load the trained state
        model.load_state_dict(state_dict)
        model.eval()

        # Generate embeddings
        with torch.no_grad():
            embeddings = model(x.to(device), edge_index.to(device)).cpu()
        # Write embeddings back to the graph
        tu.write_embeddings(gds, embeddings, write_property, list(node_to_index.keys()))
