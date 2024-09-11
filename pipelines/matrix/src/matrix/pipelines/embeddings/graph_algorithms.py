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

    def run(
        self, gds: GraphDataScience, graph: Any, model_name: str, write_property: str
    ):
        """Trains the algorithm and also writes the embeddings. Note that node2vec algorithm has no separate steps for training and inference (unlike GraphSage), thus training and inference are done via the same write function. This write function produces loss attribute which is required for plotting convergence (which is why we write the embeddings in the train topological_embedding_node)."""
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

    def predict_write(
        self, gds: GraphDataScience, graph: Any, model_name: str, write_property: str
    ):
        """Dummy function as Node2vec gets written in the 'run' step due to no separate predict_write function (unlike GraphSage)."""
        return

    def return_loss(self):
        """Return and save."""
        return self._loss
