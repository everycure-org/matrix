"""GDS Algorithms Classes."""
from abc import ABC, abstractmethod
from typing import List, Any, Optional
from graphdatascience import GraphDataScience


class GDSGraphAlgorithm(ABC):
    """Base class for Graph Algorithms, stores all arguments which can be used by child classes."""

    def __init__(
        self, embedding_dim: int = 512, random_seed: int = None, concurrency: int = 4
    ):
        """Common Attributes."""
        self._embedding_dim = embedding_dim
        self._random_seed = random_seed
        self._concurrency = concurrency

    @abstractmethod
    def run(
        self, gds: GraphDataScience, graph: Any, model_name: str, write_property: str
    ):
        """Run training base constructor."""
        pass

    @abstractmethod
    def predict_write(
        self, gds: GraphDataScience, graph: Any, model_name: str, write_property: str
    ):
        """Predict and write constructors."""
        pass

    @abstractmethod
    def return_loss(self):
        """Return loss constructor."""
        pass


class GDSGraphSage(GDSGraphAlgorithm):
    """GraphSAGE algorithm class."""

    def __init__(
        self,
        sample_sizes: List[int] = [25, 10],
        epochs: int = 10,
        learning_rate: float = 0.01,
        iterations: int = 10,
        tolerance: float = 1e-8,
        search_depth: int = 5,
        batch_size: int = 256,
        negative_sampling_weight: int = 20,
        activation_function: str = "sigmoid",
        feature_properties: str = "*",
        embedding_dim: int = 512,
        random_seed: Optional[int] = None,
        concurrency: int = 4,
    ):
        """GraphSAGE attributes."""
        super().__init__(embedding_dim, random_seed, concurrency)
        self._sample_sizes = sample_sizes
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._iterations = iterations
        self._tolerance = tolerance
        self._search_depth = search_depth
        self._batch_size = batch_size
        self._negative_sampling_weight = negative_sampling_weight
        self._activation_function = activation_function
        self._feature_properties = feature_properties
        self._loss = None

    # https://neo4j.com/docs/graph-data-science/current/machine-learning/node-embeddings/graph-sage/
    def run(
        self, gds: GraphDataScience, graph: Any, model_name: str, write_property: str
    ):
        """Train the algorithm."""
        model, attr = gds.beta.graphSage.train(
            graph,
            modelName=model_name,
            concurrency=self._concurrency,
            sampleSizes=self._sample_sizes,
            learningRate=self._learning_rate,
            maxIterations=self._iterations,
            tolerance=self._tolerance,
            embeddingDimension=self._embedding_dim,
            batchSize=self._batch_size,
            randomSeed=self._random_seed,
            epochs=self._epochs,
            searchDepth=self._search_depth,
            negativeSampleWeight=self._negative_sampling_weight,
            activationFunction=self._activation_function,
            featureProperties=self._feature_properties,
        )
        self._loss = attr["modelInfo"]["metrics"]["iterationLossesPerEpoch"][0]
        return model, attr

    def return_loss(self):
        """Return loss."""
        return self._loss

    def predict_write(
        self, gds: GraphDataScience, graph: Any, model_name: str, write_property: str
    ):
        """Predict and save."""
        model = gds.model.get(model_name)
        model.predict_write(graph, writeProperty=write_property)


class GDSNode2Vec(GDSGraphAlgorithm):
    """Node2Vec algorithm class."""

    def __init__(
        self,
        walk_length: int = 80,
        walks_per_node: int = 10,
        in_out_factor: float = 1.0,
        return_factor: float = 1.0,
        iterations: int = 10,
        negative_sampling_rate: int = 5,
        window_size: int = 10,
        initial_learning_rate: float = 0.01,
        min_learning_rate: float = 0.0001,
        embedding_dim: int = 512,
        random_seed: Optional[int] = None,
        concurrency: int = 4,
    ):
        """Node2Vec Attributes."""
        super().__init__(embedding_dim, random_seed, concurrency)
        self._walk_length = walk_length
        self._walks_per_node = walks_per_node
        self._in_out_factor = in_out_factor
        self._return_factor = return_factor
        self._iterations = iterations
        self._negative_sampling_rate = negative_sampling_rate
        self._window_size = window_size
        self._initial_learning_rate = initial_learning_rate
        self._min_learning_rate = min_learning_rate
        self._loss = None

    # https://neo4j.com/docs/graph-data-science/current/machine-learning/node-embeddings/node2vec/
    def run(
        self, gds: GraphDataScience, graph: Any, model_name: str, write_property: str
    ):
        """Train the algorithm and write."""
        attr = gds.node2vec.write(
            G=graph,
            writeProperty=write_property,
            walkLength=self._walk_length,
            walksPerNode=self._walks_per_node,
            embeddingDimension=self._embedding_dim,
            inOutFactor=self._in_out_factor,
            returnFactor=self._return_factor,
            iterations=self._iterations,
            negativeSamplingRate=self._negative_sampling_rate,
            windowSize=self._window_size,
            initialLearningRate=self._initial_learning_rate,
            minLearningRate=self._min_learning_rate,
            randomSeed=self._random_seed,
        )
        self._loss = [int(x) for x in attr["lossPerIteration"]]

    def predict_write(
        self, gds: GraphDataScience, graph: Any, model_name: str, write_property: str
    ):
        """Predict and save; dummy function as Node2Vec saves the embeddings after training."""
        return

    def return_loss(self):
        """Return and save."""
        return self._loss
