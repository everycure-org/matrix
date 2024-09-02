"""GDS Class algorithms."""
from abc import ABC


class GDSGraphAlgorithm(ABC):
    """Base class for Graph Algorithms, stores all arguments which canbe then used by child classes."""

    def __init__(
        self,
        feature_properties="*",
        relationship_types="*",
        embedding_dim=512,
        random_seed=None,
        concurrency=4,
        epochs=10,
        iterations=10,
        in_out_factor=1,
        return_factor=1,
        batch_size=256,
        tolerance=1e-8,
        search_depth=5,
        learning_rate=0.1,
        initial_learning_rate=0.01,
        min_learning_rate=0.0001,
        sample_sizes=[25, 10],
        negative_sampling_rate=5,
        walk_length=80,
        walks_per_node=10,
        window_size=10,
        negative_sampling_weight=20,
        activation_function="sigmoid",
    ):
        """Get all attributes."""
        self._embedding_dim = embedding_dim
        self._random_seed = random_seed
        self._concurrency = concurrency
        self._epochs = epochs
        self._iterations = iterations
        self._in_out_factor = in_out_factor
        self._return_factor = return_factor
        self._batch_size = batch_size
        self._tolerance = tolerance
        self._search_depth = search_depth
        self._learning_rate = learning_rate
        self._initial_learning_rate = initial_learning_rate
        self._min_learning_rate = min_learning_rate
        self._sample_sizes = sample_sizes
        self._negative_sampling_rate = negative_sampling_rate
        self._negative_sampling_weight = negative_sampling_weight
        self._walk_length = walk_length
        self._walks_per_node = walks_per_node
        self._window_size = window_size
        self._activation_function = activation_function
        self._feature_properties = feature_properties
        self._relationship_types = relationship_types
        self._loss = None


class GDSGraphSage(GDSGraphAlgorithm):
    """GraphSAGE algorithm class."""

    # https://neo4j.com/docs/graph-data-science/current/machine-learning/node-embeddings/graph-sage/
    def run(self, gds, model_name, graph, write_property):
        """Train the algorithm."""
        model, attr = gds.beta.graphSage.train(
            graph,
            modelName=model_name,
            sampleSizes=self._sample_sizes,
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

    def predict_write(self, model_name, gds, graph, write_property):
        """Predict and save."""
        model = gds.model.get(model_name)
        model.predict_write(graph, writeProperty=write_property)


class GDSNode2Vec(GDSGraphAlgorithm):
    """Node2Vec algorithm class."""

    # https://neo4j.com/docs/graph-data-science/current/machine-learning/node-embeddings/node2vec/
    def run(self, gds, graph, model_name, write_property):
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

    def predict_write(self, model_name, gds, graph, write_property):
        """Predict and save; dummy function as Node2Vec saves the embeddings after training."""
        return

    def return_loss(self):
        """Return and save."""
        return self._loss
