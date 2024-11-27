from neo4j import AsyncGraphDatabase

from kedro.pipeline import Pipeline, node, pipeline

import pandas as pd


async def fetch_data_as_dataframe(query):
    """
    Executes a Cypher query asynchronously in a read transaction and
    returns the results as a Pandas DataFrame.

    :param query: The Cypher query to execute.
    :param parameters: Optional dictionary of query parameters.
    :return: A Pandas DataFrame containing the query results.
    """
    # Initialize the Neo4j driver
    driver = AsyncGraphDatabase.driver("bolt://127.0.0.1:7687", auth=("neo4j", "admin"))

    async def run_query(tx):
        # Executes the query within the transaction
        result = await tx.run(query)
        # Convert results to a list of dictionaries
        records = []
        async for record in result:
            records.append(record.data())
        return records

    try:
        # Async session
        async with driver.session(database="moa") as session:
            # Use execute_read to ensure the query runs in a read transaction
            records = await session.execute_read(run_query)
            # Create a DataFrame from the list of dictionaries
            dataframe = pd.DataFrame(records)
            return dataframe.astype(str)
    finally:
        await driver.close()


def generate_paths(ground_truth: pd.DataFrame):
    shards = {}
    for index, row in ground_truth.iterrows():
        # TODO: Should we group by source and use single query per source? or does that make it unbalanced?

        # NOTE: Setting quick limit
        if index == 1000:
            break

        # Render query
        query = f"""
            MATCH p=(drug:Entity {{id: '{row['source']}'}})-[*2..2]->(disease:Entity {{id: '{row['target']}'}}) 
            WITH [node IN nodes(p) | node.id] as nodes, [node in nodes(p) | labels(node)[1]] as labels, [rel in relationships(p) | type(rel)] as rels 
            RETURN nodes, labels, rels"""

        # Invoke function
        # TODO: We need to optimize parallelism, whats number of queries we should execute in parallel for best result?
        shard_path = f"idx={index}/shard"
        shards[shard_path] = lambda query=query: fetch_data_as_dataframe(query)

    return shards


def print_(df):
    breakpoint()


def create_pipeline(**kwargs) -> Pipeline:
    """Create moa pipeline."""
    return pipeline(
        [
            node(
                func=generate_paths,
                inputs=["modelling.raw.ground_truth.positives@pandas"],
                outputs="moa.int.paths@partitioned",
                name="generate_paths",
            ),
            # node(
            #     func=print_,
            #     inputs=["moa.int.paths@spark"],
            #     outputs=None,
            #     name="show",
            # ),
        ]
    )
