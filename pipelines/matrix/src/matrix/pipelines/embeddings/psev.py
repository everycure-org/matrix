import ensmallen
import tqdm
import numpy as np
import anndata
import pandas as pd
from scipy import sparse
import pyspark.sql.functions as F
from pyspark.sql.dataframe import DataFrame
from pyspark.ml.linalg import SparseVector
from pyspark.sql import SparkSession
from refit.v1.core.unpack import unpack_params
from matrix.datasets.ensmallen import NODE_ID, NODE_TYPE


import logging

logger = logging.getLogger(__name__)


def csr_to_pyspark(sparse_matrix, row_ids):
    """turns a sparse matrix from scipy into pyspark
    usig the row_ids

    returns a list of tuples: (id, SparseVector)
    """
    nrows, ncols = sparse_matrix.shape
    assert nrows == len(row_ids)
    assert isinstance(sparse_matrix, sparse._csr.csr_matrix)

    data = [
        (the_id, SparseVector(ncols, sparse_matrix[i].indices, sparse_matrix[i].data))
        for i, the_id in tqdm.tqdm(enumerate(row_ids), total=len(row_ids))
    ]
    return data


def annotate_to_pyspark_nodes(adata_ppv: anndata.AnnData, nodes_df: DataFrame):
    """
    adding the embedding to a pyspark.Dataframe, based on node_id

    adata_ppv = catalog.load('ppv_genegene')
    nodes_df = catalog.load("spoke_grape_genegene_augmented_pyspark_nodes")
    """
    # assert adata_ppv.shape[0] == nodes_df.count(), "adata and pyspark-frame have different number of nodes"

    disease_ix = adata_ppv.var.index.map(lambda x: x.startswith("DOID:"))
    assert sum(disease_ix) > 0, "no disease in embedding dimension"
    the_matrix = sparse.csr_matrix(adata_ppv.X[:, disease_ix])

    data = csr_to_pyspark(the_matrix, adata_ppv.obs_names)
    spark_session = SparkSession.builder.getOrCreate()
    dataframe = spark_session.createDataFrame(data, [NODE_ID, "sparse_embedding"])

    count_before = nodes_df.count()
    nodes_df = nodes_df.join(dataframe, on=NODE_ID, how="inner")
    count_after = nodes_df.count()

    assert count_before == count_after, f"lost some nodes {count_before}->{count_after}"
    return nodes_df


def generate_psev_weights(nodes_df: DataFrame):
    """just make up some psev weights,
    i.e. a vector per disease that weights spoke nodes
    """
    n_diseases = 100  # TODO hardcoded for now
    disease_ids = nodes_df.filter(F.col(NODE_TYPE) == "Disease").select(F.col(NODE_ID)).toPandas()[NODE_ID].values
    disease_ids = disease_ids[:n_diseases]

    all_ids = (
        nodes_df
        # exclude compounds as so many are singletons
        .filter(F.col(NODE_TYPE) != "Compound")
        .select(F.col(NODE_ID))
        .toPandas()[NODE_ID]
        .values
    )

    vectors = []
    for i in tqdm.trange(len(disease_ids)):
        # how many terms are associated with the ddisease
        # lets be quite generous here, to make sure we
        # actually have some nodes in the subgraphs

        # pick number of terms and spread a number of patietns across them
        n_terms = np.random.choice(range(20, 100))
        n_patient = np.random.choice(range(100, 10000))

        # pick the terms randomly
        term_ix = np.random.choice(len(all_ids), size=n_terms, replace=False)
        term_weights = np.random.dirichlet(alpha=0.5 * np.ones(n_terms))
        term_counts = np.random.multinomial(n=n_patient, pvals=term_weights)

        v = sparse.csr_matrix((term_counts, (np.zeros_like(term_ix), term_ix)), shape=(1, len(all_ids)))
        vectors.append(v)

    X = sparse.vstack(vectors)

    return anndata.AnnData(X, obs=pd.DataFrame(index=disease_ids), var=pd.DataFrame(index=all_ids))


@unpack_params()
def estimate_psev(
    G: ensmallen.Graph, node_weights: anndata.AnnData, alpha: float, iterations: int, max_walk_length: int
):
    logger.info(
        f"running PSEVs on graph (nodes: {G.get_number_of_nodes()} edges: {G.get_number_of_edges()}); alpha:{alpha}, iter:{iterations}, max_len: {max_walk_length}"
    )

    assert (
        len(set(G.get_node_names()) & set(node_weights.var_names)) > 10000
    ), "minimal overlap between weights and graph. are we using the right stage of preprocessing??"

    def extract_weighting_dict(index):
        """get the weights out of the adata (a row)
        and label the weights by its nodename
        """
        weights = node_weights[index, :].X
        int_to_nodename = {index: n for index, n in enumerate(node_weights.var_names)}
        row_ix, col_ix, data = sparse.find(weights)
        weight_dict = {int_to_nodename[node]: w for node, w in zip(col_ix, data)}
        return weight_dict

    diseases = np.sort(node_weights.obs_names).tolist()
    nodenames = np.sort(G.get_node_names()).tolist()

    disease_to_index = {d: i for i, d in enumerate(diseases)}
    nodenames_to_index = {n: i for i, n in enumerate(nodenames)}

    # sparse matrix for the PSEVs in row/col format
    matrix_builder = CSR_builder_ijk(n_rows=len(diseases), n_cols=len(nodenames), dtype="float")

    for i in tqdm.trange(node_weights.shape[0]):  # iter through all diseases
        the_disease = node_weights.obs_names[i]
        weight_dict = extract_weighting_dict(i)
        # fitler out anythign not in the graph
        weight_dict = {name: weight for name, weight in weight_dict.items() if G.has_node_name(name)}

        if len(weight_dict) > 0:
            nodename_to_psev = G.psev_estimation_nodenames(weight_dict, alpha, iterations, max_walk_length)

            ## into matrix format
            row_ix = disease_to_index[the_disease]
            for name, value in nodename_to_psev.items():
                matrix_builder.add(row_ix, nodenames_to_index[name], value)

    X = matrix_builder.build()

    # note: transpose to get it into an "embedding"-shape, i.e. #nodes x embedding dimension
    PSEV = anndata.AnnData(X, obs=pd.DataFrame(index=diseases), var=pd.DataFrame(index=nodenames)).T
    return PSEV


class CSR_builder_ijk:
    """Iteratively build a CSR-matrix from single datapoints"""

    def __init__(self, n_rows, n_cols, dtype):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.dtype = dtype
        self.ii = []
        self.jj = []
        self.data = []

    def add(self, row_ix: int, col_ix: int, data):
        assert row_ix < self.n_rows, f"cant add to row {row_ix}, max {self.n_rows} rows allowed"
        assert col_ix < self.n_cols, f"cant add to col {col_ix}, max {self.n_cols} cols allowed"
        self.ii.append(row_ix)
        self.jj.append(col_ix)
        self.data.append(data)

    def build(self):
        return sparse.csr_matrix((self.data, (self.ii, self.jj)), shape=[self.n_rows, self.n_cols])


class CSR_builder:
    """Iteratively build a CSR-matrix, adding one row after another"""

    def __init__(self, n_cols, dtype):
        self.n_cols = n_cols
        self.dtype = dtype
        self.indptr = [0]
        self.indices = []
        self.data = []

    def add(self, col_ix: list, data: list):
        assert len(col_ix) == len(data)
        if len(col_ix) > 0:  # silly, np.max([]) fails
            assert np.max(col_ix) < self.n_cols, f"cant add col {np.max(col_ix)} to a {self.n_cols}-col matrix"

        # TODO check if dtype is satiesfied
        # TODO check that col_ix is sorteD?
        self.indices.extend(col_ix)
        self.data.extend(data)
        self.indptr.append(self.indptr[-1] + len(data))

    def build(self):
        nrows = (
            len(self.indptr) - 1
        )  # indptr is one longer than the actual rows, carries the #elements in the last index
        return sparse.csr_matrix((self.data, self.indices, self.indptr), shape=(nrows, self.n_cols), dtype=self.dtype)


def test_csr_builder_ijk():
    b = CSR_builder_ijk(n_rows=2, n_cols=4, dtype="int32")
    b.add(0, 0, -1)
    b.add(1, 3, -1)

    x = b.build().toarray()
    expected = np.array([[-1, 0, 0, 0], [0, 0, 0, -1]])

    assert np.all(x == expected)


def test_csr_builder():
    b = CSR_builder(n_cols=4, dtype="int32")
    b.add([1, 2], [-1, -1])

    b.add([0], [-1])
    b.add([0, 1, 2], [-1, 1, 1])
    b.add([0], [-1])
    b.add([], [])  # empty row
    x = b.build().toarray()
    expected = np.array([[0, -1, -1, 0], [-1, 0, 0, 0], [-1, 1, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 0]])

    assert np.all(x == expected)
