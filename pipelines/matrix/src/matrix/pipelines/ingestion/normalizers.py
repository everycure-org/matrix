"""Node Normalization for Ingestion Pipeline."""
import os
import logging
import requests
import time
from robokop_genetics.genetics_normalization import GeneticsNormalizer

NORMALIZATION_CODE_VERSION = '1.2'
NODE_TYPES = 'category'
NAMED_THING = 'biolink:NamedThing'
SYNONYMS = 'equivalent_identifiers'
INFORMATION_CONTENT = 'information_content'
DESCRIPTION = 'description'
PREDICATE = 'predicate'

# node property name for node types that did not normalize
CUSTOM_NODE_TYPES = 'custom_node_types'

# predicate to use when normalization fails
FALLBACK_EDGE_PREDICATE = 'biolink:related_to'


class NodeNormalizer:
    """
    Class that contains methods relating to node normalization of KGX data.

    the input node list should be KGX compliant and have the following columns that may be
    changed during the normalization:

        id: the id value to be normalized upon
        name: the name of the node
        category: the semantic type(s)
        equivalent_identifiers: the list of synonymous ids
    """

    def __init__(self,
                 node_normalization_version: str = 'latest',
                 biolink_version: str = 'latest',
                 strict_normalization: bool = True,
                 conflate_node_types: bool = False):
        """
        constructor
        :param node_normalization_version - not implemented yet
        """
        self.logger = logging.getLogger(__name__)

        # storage for regular nodes that failed to normalize
        self.failed_to_normalize_ids = set()
        # storage for variant nodes that failed to normalize
        self.failed_to_normalize_variant_ids = {}
        # flag that determines whether nodes that failed to normalize should be included or thrown out
        self.strict_normalization = strict_normalization
        self.biolink_version = biolink_version
        self.biolink_compliant_node_types = None
        # whether the normalizer should conflate node types (ie combine genes and proteins)
        self.conflate_node_types = conflate_node_types
        # storage for variant nodes that split into multiple new nodes in normalization
        self.variant_node_splits = {}
        # normalization map for future look up of all normalized node IDs
        self.node_normalization_lookup = {}
        self.node_norm_endpoint = os.getenv('NODE_NORMALIZATION_ENDPOINT', 'https://nodenormalization-sri.renci.org/')
        self.sequence_variant_normalizer = None
        self.variant_node_types = None

    def hit_node_norm_service(self, curies, retries=0):
        """
            call the NN service
            :param curies
            :param retries
        """

        resp: requests.models.Response = requests.post(f'{self.node_norm_endpoint}get_normalized_nodes',
                                                       json={'curies': curies,
                                                             'conflate': self.conflate_node_types,
                                                             'drug_chemical_conflate': self.conflate_node_types,
                                                             'description': True})
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
                    self.logger.error(error_message)
                    resp.raise_for_status()
                else:
                    error_message += f', retrying.. (attempt {retries})'
                    time.sleep(retries * 3)
                    self.logger.error(error_message)
                    return self.hit_node_norm_service(curies, retries)
            else:
                # we should never get a legitimate 4xx response from node norm,
                # crash with an error for troubleshooting
                if resp.status_code == 422:
                    error_message += f'(curies: {curies})'
                self.logger.error(error_message)
                resp.raise_for_status()

    def normalize_node_data(self, node_list: list, block_size: int = 1000) -> list:
        """
        This method calls the NodeNormalization web service to get the normalized identifier and name of the node.
        the data comes in as a node list.

        :param node_list: A list with items to normalize
        :param block_size: the number of curies in the request

        :return:
        """

        self.logger.debug(f'Start of normalize_node_data. items: {len(node_list)}')

        # init the cache - this accumulates all the results from the node norm service
        cached_node_norms: dict = {}

        # create a unique set of node ids
        tmp_normalize: set = set([node['id'] for node in node_list])

        # convert the set to a list so we can iterate through it
        to_normalize: list = list(tmp_normalize)

        # init the array index lower boundary
        start_index: int = 0

        # get the last index of the list
        last_index: int = len(to_normalize)

        self.logger.debug(f'{last_index} unique nodes found in this group.')

        # grab chunks of the data frame
        while True:
            if start_index < last_index:
                # define the end index of the slice
                end_index: int = start_index + block_size

                # force the end index to be the last index to insure no overflow
                if end_index >= last_index:
                    end_index = last_index

                self.logger.debug(f'Working block {start_index} to {end_index}.')

                # collect a slice of records from the data frame
                data_chunk: list = to_normalize[start_index: end_index]

                # hit the node norm api
                normalization_json = self.hit_node_norm_service(curies=data_chunk)
                if normalization_json:
                    # merge the normalization results with what we have gotten so far
                    cached_node_norms.update(**normalization_json)
                else:
                    # this shouldn't happen but if the API returns an empty dict instead of nulls,
                    # assume none of the curies normalize
                    empty_responses = {curie: None for curie in data_chunk}
                    cached_node_norms.update(empty_responses)

                # move on down the list
                start_index += block_size
            else:
                break

        # reset the node index
        node_idx = 0

        # node ids that failed to normalize
        failed_to_normalize: list = []

        # look up valid node types if needed
        if not self.strict_normalization and not self.biolink_compliant_node_types:
            biolink_lookup = EdgeNormalizer(edge_normalization_version=self.biolink_version)
            self.biolink_compliant_node_types = biolink_lookup.get_valid_node_types()

        # for each node update the node with normalized information
        # store the normalized IDs in self.node_normalization_lookup for later look up
        while node_idx < len(node_list):

            # get the next node list item by index
            current_node = node_list[node_idx]
            current_node_id = current_node['id']

            # make sure there is a name
            if 'name' not in current_node or not current_node['name']:
                current_node['name'] = current_node['id'].split(':')[-1]

            # remove properties with null values, remove newline characters
            for key in list(current_node.keys()):
                value = current_node[key]
                if value is None:
                    del (current_node[key])
                else:
                    if isinstance(value, str):
                        current_node[key] = value.replace("\n", "")

            # if strict normalization is off, enforce valid node types
            if not self.strict_normalization:

                if NODE_TYPES not in current_node:
                    current_node[NODE_TYPES] = [NAMED_THING]

                # remove all the bad types and make them a property instead
                invalid_node_types = [node_type for node_type in current_node[NODE_TYPES] if
                                      node_type not in self.biolink_compliant_node_types]
                if invalid_node_types:
                    current_node[CUSTOM_NODE_TYPES] = invalid_node_types

                # keep all the valid types
                current_node[NODE_TYPES] = [node_type for node_type in current_node[NODE_TYPES] if
                                            node_type in self.biolink_compliant_node_types]
                # add the NAMED_THING type if it's not there
                if NAMED_THING not in current_node[NODE_TYPES]:
                    current_node[NODE_TYPES].append(NAMED_THING)

                # enforce that the list is really a set
                current_node[NODE_TYPES] = list(set(current_node[NODE_TYPES]))

            # did we get a response from the normalizer
            current_node_normalization = cached_node_norms[current_node_id]
            if current_node_normalization is not None:

                current_node_id_section = current_node_normalization['id']

                # update the node with the normalized info
                normalized_id = current_node_id_section['identifier']
                current_node['id'] = normalized_id
                current_node[NODE_TYPES] = current_node_normalization['type']
                current_node[SYNONYMS] = list(item['identifier'] for item in current_node_normalization[SYNONYMS])

                # set the name as the primary label if it exists
                if 'label' in current_node_id_section:
                    current_node['name'] = current_node_id_section['label']

                # set the node description and/or information content if they are present
                if 'information_content' in current_node_normalization:
                    current_node[INFORMATION_CONTENT] = current_node_normalization[INFORMATION_CONTENT]
                if 'description' in current_node_id_section:
                    current_node[DESCRIPTION] = current_node_id_section['description']

                self.node_normalization_lookup[current_node_id] = [normalized_id]
            else:
                # we didn't find a normalization - add it to the failure list
                failed_to_normalize.append(current_node_id)
                if self.strict_normalization:
                    # if strict normalization is on we set that index to None so that the node is removed
                    node_list[node_idx] = None
                    # store None in the normalization map so we know it didn't normalize
                    self.node_normalization_lookup[current_node_id] = None
                else:
                    # if strict normalization is off keep it and set its previous id in the normalization map
                    self.node_normalization_lookup[current_node_id] = [current_node_id]

            # go to the next node index
            node_idx += 1

        # if something failed to normalize - log it and optionally remove it from the node list
        if len(failed_to_normalize) > 0:
            self.failed_to_normalize_ids.update(failed_to_normalize)

            # if strict remove all nodes that failed normalization
            if self.strict_normalization:
                node_list[:] = [d for d in node_list if d is not None]

        self.logger.debug(f'End of normalize_node_data.')

        # return the failed list to the caller
        return failed_to_normalize

    def normalize_sequence_variants(self, variant_nodes: list):
        """
        Normalize Genetic Sequence Variants

        :param variant_nodes:
        """

        if not variant_nodes:
            return

        if not self.sequence_variant_normalizer:
            self.sequence_variant_normalizer = GeneticsNormalizer(use_cache=False)
            self.variant_node_types = self.sequence_variant_normalizer.get_sequence_variant_node_types()
        variant_node_types = self.variant_node_types

        variant_ids = [node['id'] for node in variant_nodes]
        variant_nodes.clear()

        sequence_variant_norms = self.sequence_variant_normalizer.normalize_variants(variant_ids)
        for variant_id, normalization_response in sequence_variant_norms.items():
            for normalization_info in normalization_response:
                # if the normalization info contains an ID it was a success
                if 'id' in normalization_info:
                    normalized_node = {
                        'id': normalization_info["id"],
                        'name': normalization_info["name"],
                        # as long as sequence variant types are all the same we can skip this assignment
                        # 'category': normalized_info["type"],
                        'category': variant_node_types,
                        'equivalent_identifiers': normalization_info["equivalent_identifiers"]
                    }
                    variant_nodes.append(normalized_node)
                    # assume we don't have a split and store the id for look up
                    self.node_normalization_lookup[variant_id] = [normalization_info["id"]]
                else:
                    # otherwise an error occurred
                    error_for_logs = f'{normalization_info["error_type"]}: {normalization_info["error_message"]}'
                    self.failed_to_normalize_variant_ids[variant_id] = error_for_logs
                    if self.strict_normalization:
                        self.node_normalization_lookup[variant_id] = None
                    else:
                        self.node_normalization_lookup[variant_id] = [variant_id]
                        # TODO for now we dont preserve other properties on variant nodes that didnt normalize
                        # the splitting makes that complicated and doesnt seem worth it until we have a good use case
                        fake_normalized_node = {
                            'id': variant_id,
                            'name': variant_id,
                            'category': variant_node_types,
                            'equivalent_identifiers': []
                        }
                        variant_nodes.append(fake_normalized_node)
            if len(normalization_response) > 1:
                # if we have more than one response here assume its a split variant and no errors
                split_ids = [node['id'] for node in normalization_response]
                self.variant_node_splits[variant_id] = split_ids
                # this will overwrite the previous single IDs stored
                self.node_normalization_lookup[variant_id] = split_ids

        return variant_nodes

    def get_current_node_norm_version(self):
        """
        Retrieves the current production version from the node normalization service.
        """
        # fetch the node norm openapi spec
        node_norm_openapi_url = f'{self.node_norm_endpoint}openapi.json'
        resp: requests.models.Response = requests.get(node_norm_openapi_url)

        # did we get a good status code
        if resp.status_code == 200:
            # convert json to dict
            openapi: dict = resp.json()
            # extract the version
            node_norm_version = openapi['info']['version']
            return node_norm_version
        else:
            # this shouldn't happen, raise an exception
            resp.raise_for_status()


class EdgeNormalizationResult:
    def __init__(self,
                 predicate: str,
                 inverted: bool = False,
                 properties: dict = None):
        self.predicate = predicate
        self.inverted = inverted
        self.properties = properties


class EdgeNormalizer:
    """
    Class that contains methods relating to edge normalization.
    """

    def __init__(self, edge_normalization_version: str = 'latest'):
        # create a logger
        self.logger = logging.getLogger(__name__)
        # normalization map for future look up of all normalized predicates
        self.edge_normalization_lookup = {}
        self.cached_edge_norms = {}
        self.edge_norm_endpoint = os.getenv('EDGE_NORMALIZATION_ENDPOINT', 'https://bl-lookup-sri.renci.org/')

        if edge_normalization_version != 'latest':
            if self.check_bl_version_valid(edge_normalization_version):
                self.edge_norm_version = edge_normalization_version
            else:
                raise requests.exceptions.HTTPError(f'Edge norm version {edge_normalization_version} '
                                                    f'is not supported by endpoint {self.edge_norm_endpoint}.')
        else:
            self.edge_norm_version = self.get_current_edge_norm_version()

    def normalize_edge_data(self, edge_list: list, block_size: int = 100) -> list:
        """
        This method calls the EdgeNormalization web service to retrieve information for normalizing edges.

        :param edge_list: A list of edges to normalize - edges are dictionaries with the PREDICATE constant as a key
        :param block_size: the number of predicates to process in a single call
        :return:
        """

        # find the predicates that have not been normalized yet
        predicates_to_normalize = set()
        for edge in edge_list:
            if edge[PREDICATE] not in self.edge_normalization_lookup:
                predicates_to_normalize.add(edge[PREDICATE])

        # convert the set to a list so we can iterate through it
        predicates_to_normalize_list = list(predicates_to_normalize)

        # dictionary to accumulate normalization results
        edge_normalizations = {}

        # indexes for iterating through the list in chunks
        start_index: int = 0
        last_index: int = len(predicates_to_normalize_list)
        while True:
            if start_index >= last_index:
                # no more predicates to process, break the loop
                break

            # define the end index of the slice
            end_index: int = start_index + block_size

            # force the end index to be at most the last index to ensure no overflow
            if end_index > last_index:
                end_index = last_index

            # collect a slice of predicates
            predicate_chunk: list = predicates_to_normalize_list[start_index: end_index]

            # hit the edge normalization service
            request_url = f'{self.edge_norm_endpoint}resolve_predicate?version={self.edge_norm_version}&predicate='
            request_url += '&predicate='.join(predicate_chunk)
            self.logger.debug(f'Sending request: {request_url}')
            resp: requests.models.Response = requests.get(request_url)

            # if we get a success status code
            if resp.status_code == 200:
                # merge the response with what we have already
                rvs: dict = resp.json()
                edge_normalizations.update(**rvs)
            elif resp.status_code == 404:
                # this should not happen but if it does fail gracefully and assume no meaningful normalization results
                # (current versions of bl look up should always return at least a default for each predicate)
                pass
            else:
                # this is a real error with the edge normalizer so we bail
                error_message = f'Edge norm response code: {resp.status_code}'
                self.logger.error(error_message)
                resp.raise_for_status()

            # move on down the list
            start_index += block_size

        # storage for items that failed to normalize
        failed_to_normalize: list = list()

        # walk through the unique predicates and process normalized predicates for the lookup map
        for predicate in predicates_to_normalize:
            # did the service return a value with an identifier
            if predicate in edge_normalizations and \
                    (('predicate' in edge_normalizations[predicate]) or \
                     ('identifier' in edge_normalizations[predicate])):
                normalization_info = edge_normalizations[predicate]
                if 'predicate' in normalization_info:
                    normalized_predicate = normalization_info.pop('predicate')
                else:
                    normalized_predicate = normalization_info.pop('identifier')
                normalization_info.pop('label', None)  # just deleting this key, it's not needed anymore
                inverted = True if normalization_info.pop('inverted', False) else False
                self.edge_normalization_lookup[predicate] = EdgeNormalizationResult(predicate=normalized_predicate,
                                                                                    inverted=inverted,
                                                                                    properties=normalization_info)
            else:
                # this should not happen but if it does use the fallback predicate
                self.edge_normalization_lookup[predicate] = EdgeNormalizationResult(predicate=FALLBACK_EDGE_PREDICATE)
                failed_to_normalize.append(predicate)

        # if something failed to normalize output it
        # if failed_to_normalize:
        #    self.logger.error(f'Failed to normalize: {", ".join(failed_to_normalize)}')

        # return the failed list to the caller
        return failed_to_normalize

    def get_current_edge_norm_version(self):
        """
        Retrieves the current production version from the edge normalization service.
        """
        versions = self.get_available_versions()
        return versions[0]

    def check_bl_version_valid(self, bl_version: str):
        """
        Checks if the requested version is supported by the API.
        """
        if bl_version in self.get_available_versions():
            return True
        else:
            return False

    def get_available_versions(self):
        """
        Get available versions.
        """
        # call the versions endpoint
        edge_norm_versions_url = f'{self.edge_norm_endpoint}versions'
        resp: requests.models.Response = requests.get(edge_norm_versions_url)

        # did we get a good status code
        if resp.status_code == 200:
            # parse json
            versions = resp.json()
            return versions  # array of versions
        else:
            # this shouldn't happen, raise an exception
            resp.raise_for_status()

    def check_node_type_valid(self, node_type: str):
        """
        Check Node Type Validity.
        """
        if node_type in self.get_valid_node_types():
            return True
        else:
            return False

    def get_valid_node_types(self):
        """
        Gets valid node types.
        """
        # call the descendants endpoint with the root node type
        edge_norm_descendants_url = f'{self.edge_norm_endpoint}bl/{NAMED_THING}/descendants?version={self.edge_norm_version}'
        resp: requests.models.Response = requests.get(edge_norm_descendants_url)

        # did we get a good status code
        if resp.status_code == 200:
            # parse json
            descendants = resp.json()
            return descendants  # array of descendants
        else:
            # this shouldn't happen, raise an exception
            resp.raise_for_status()
