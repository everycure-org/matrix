import argparse
import pathlib
import time
import pandas as pd
import requests
from dotenv import load_dotenv
import logging
import os
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[ logging.StreamHandler(sys.stdout) ]
)

class NodeNormalizer:

    def __init__(self, robokop_nodes: pathlib.Path, robokop_edges: pathlib.Path, rtx_kg2_nodes: pathlib.Path, rtx_kg2_edges: pathlib.Path):
        load_dotenv()
        self.robokop_nodes = robokop_nodes
        self.robokop_edges = robokop_edges
        self.rtx_kg2_nodes = rtx_kg2_nodes
        self.rtx_kg2_edges = rtx_kg2_edges
        self.nn_endpoint = f'{os.getenv("NODE_NORMALIZER_SERVER", "https://nodenormalization-sri.renci.org/")}/get_normalized_nodes'

    def __call__(self):


        robokop_nodes_df = pd.read_csv(self.robokop_nodes, sep="\t")
        rtx_kg2_nodes_df = pd.read_csv(self.rtx_kg2_nodes, sep="\t")

        rtx_kg2_nodes_df = rtx_kg2_nodes_df.rename(columns={"equivalent_curies": "equivalent_identifiers"})

        logger.debug(robokop_nodes_df.columns)
        logger.debug(rtx_kg2_nodes_df.columns)

        robokop_edges_df = pd.read_csv(self.robokop_edges, sep="\t")
        rtx_kg2_edges_df = pd.read_csv(self.rtx_kg2_edges, sep="\t")

        rtx_kg2_edges_df = rtx_kg2_edges_df.rename(columns={"knowledge_source": "primary_knowledge_source"})
        # id looks like a sequence/index left over from an export that should have excluded it...removing now
        del rtx_kg2_edges_df['id']

        logger.debug(robokop_edges_df.columns)
        logger.debug(rtx_kg2_edges_df.columns)

        self.normalize_dataframe(robokop_nodes_df, robokop_edges_df)
        self.normalize_dataframe(rtx_kg2_nodes_df, rtx_kg2_edges_df)

        # key off these columns for edges: "subject", "predicate", "object", "primary_knowledge_source"
        joined_edges = robokop_edges_df.merge(rtx_kg2_edges_df, on=["subject", "predicate", "object", "primary_knowledge_source"], how="outer", suffixes=('_robokop', '_rtx'))
        # print(joined_edges.shape)
        # joined_edges.to_csv(pathlib.Path("/tmp/merged_edges.tsv"), sep="\t")
        deduped_joined_edges = joined_edges.drop_duplicates()
        # deduped_joined_edges.to_csv(pathlib.Path("/tmp/merged_edges_deduped.tsv"), sep="\t")

        # key off these columns for nodes: "id", "category"
        joined_nodes = robokop_nodes_df.merge(rtx_kg2_nodes_df, on=["id", "category"], how="outer", suffixes=('_robokop', '_rtx'))
        # print(joined_nodes.shape)
        # joined_nodes.to_csv(pathlib.Path("/tmp/merged_nodes.tsv"), sep="\t")
        deduped_joined_nodes = joined_nodes.drop_duplicates()
        # deduped_joined_nodes.to_csv(pathlib.Path("/tmp/merged_nodes_deduped.tsv"), sep="\t")


    def normalize_dataframe(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame):

        for idx, row in edges_df.iterrows():

            sub = row['subject']
            obj = row['object']

            nn_json_response = self.hit_node_norm_service(curies=[sub, obj])

            if nn_json_response is None:
                continue

            if nn_json_response[sub] is not None:
                nodes_df.loc[nodes_df.id == sub, 'id'] = nn_json_response[sub]['id']['identifier']
                edges_df.loc[edges_df.subject == sub, 'subject'] = nn_json_response[sub]['id']['identifier']

            if nn_json_response[obj] is not None:
                nodes_df.loc[nodes_df.id == obj, 'id'] = nn_json_response[obj]['id']['identifier']
                edges_df.loc[edges_df.object == obj, 'object'] = nn_json_response[obj]['id']['identifier']


    def hit_node_norm_service(self, curies, retries=0):

        request_json = {'curies': curies,
                        'conflate': f'{os.getenv('CONFLATE_NODE_TYPES', "true")}',
                        'drug_chemical_conflate': f'{os.getenv('CONFLATE_NODE_TYPES', "true")}',
                        'description': "true"}

        logger.debug(request_json)
        resp: requests.models.Response = requests.post(url=self.nn_endpoint, json=request_json)
        logger.debug(resp.json())

        if resp.status_code == 200:
            # if successful return the json as an object
            return resp.json()
        else:
            error_message = f'Node norm response code: {resp.status_code}'
            if resp.status_code >= 500:
                # if 5xx retry 3 times
                retries += 1
                if retries == 4:
                    error_message += ', retried 3 times, giving up..'
                    logger.error(error_message)
                    resp.raise_for_status()
                else:
                    error_message += f', retrying.. (attempt {retries})'
                    time.sleep(retries * 3)
                    logger.error(error_message)
                    return self.hit_node_norm_service(curies, retries)
            else:
                # we should never get a legitimate 4xx response from node norm,
                # crash with an error for troubleshooting
                if resp.status_code == 422:
                    error_message += f'(curies: {curies})'
                logger.error(error_message)
                resp.raise_for_status()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='perform node normalization and merging of data')
    parser.add_argument('-a', '--robokop_nodes', required=True, type=pathlib.Path, help='TSV export of ROBOKOP nodes')
    parser.add_argument('-b', '--robokop_edges', required=True, type=pathlib.Path, help='TSV export of ROBOKOP edges')
    parser.add_argument('-c', '--rtx_kg2_nodes', required=True, type=pathlib.Path, help='TSV export of RTX KG2 nodes')
    parser.add_argument('-d', '--rtx_kg2_edges', required=True, type=pathlib.Path, help='TSV export of RTX KG2 edges')

    args = parser.parse_args()
    node_normalizer = NodeNormalizer(args.robokop_nodes, args.robokop_edges, args.rtx_kg2_nodes, args.rtx_kg2_edges)
    node_normalizer()

