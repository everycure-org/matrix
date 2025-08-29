#!/usr/bin/python3
## This code is created by RTX team (https://github.com/RTXteam/RTX/) used in the version 2.7.3 of RTX-KG2c (Wood et al. 2021 doi: 10.1101/2021.10.17.464747)
"""An interface to the SRI Node and Edge Normalizer https://nodenormalization-sri.renci.org/apidocs/
e.g.:  https://nodenormalization-sri.renci.org/get_normalized_nodes?curie=CHEMBL.COMPOUND:CHEMBL76729
Report problems here: https://github.com/TranslatorSRI/NodeNormalization/issues
"""

import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


import json
import os
import pickle
import platform
import re
import shelve
import time

import requests
import requests_cache


# Class that provides a simple interface to the SRI Node normalizer
class SriNodeNormalizer:
    # Constructor
    def __init__(self):
        requests_cache.install_cache("sri_node_normalizer_requests_cache")

        self.storage_mode = "dict"  # either dict or shelve

        self.supported_types = None
        self.supported_prefixes = None
        self.cache = {}
        self.stats = {}

        if self.storage_mode == "shelve":
            self.cache = shelve.open("sri_node_normalizer_curie_cache.shelve")

        # Translation table of different curie prefixes ARAX -> normalizer
        self.curie_prefix_tx_arax2sri = {
            #'REACT': 'Reactome',
            #'Orphanet': 'ORPHANET',
            #'ICD10': 'ICD-10',
        }
        self.curie_prefix_tx_sri2arax = {
            #'Reactome': 'REACT',
            #'ORPHANET': 'Orphanet',
            #'ICD-10': 'ICD10',
        }

    # ############################################################################################
    # Store the cache of all normalizer results
    def store_cache(self):
        if self.storage_mode == "shelve":
            self.cache.sync()
            return
        if self.cache is None:
            return
        filename = "sri_node_normalizer_curie_cache.pickle"
        print(f"INFO: Storing SRI normalizer cache to {filename}")
        with open(filename, "wb") as outfile:
            pickle.dump(self.cache, outfile)

    # ############################################################################################
    # Load the cache of all normalizer results
    def load_cache(self):
        if self.storage_mode == "shelve":
            return
        filename = "sri_node_normalizer_curie_cache.pickle"
        if os.path.exists(filename):
            print(f"INFO: Reading SRI normalizer cache from {filename}")
            with open(filename, "rb") as infile:
                self.cache = pickle.load(infile)
                # self.cache = self.cache['ids']
        else:
            print(f"INFO: SRI node normalizer cache {filename} does not yet exist. Need to fill it.")

    # ############################################################################################
    # Fill the cache with KG nodes
    def fill_cache(self):
        # Get a hash of curie prefixes supported
        self.get_supported_prefixes()

        filename = "kg2_node_info.tsv"
        filesize = os.path.getsize(filename)

        # Correction for Windows line endings
        extra_bytes = 0
        fh = open(filename, "rb")
        if platform.system() == "Windows" and b"\r\n" in fh.read():
            print("WARNING: Windows line ending requires bytes_read compenstation")
            extra_bytes = 1
        fh.close()

        # Open the file and read in the curies
        fh = open(filename, "r", encoding="latin-1", errors="replace")
        print(f"INFO: Reading {filename} to pre-fill the normalizer cache")
        previous_percentage = -1
        line_counter = 0
        supported_curies = 0
        bytes_read = 0

        # Create dicts to hold all the information
        batch = []

        # Loop over each line in the file
        for line in fh:
            bytes_read += len(line) + extra_bytes

            match = re.match(r"^\s*$", line)
            if match:
                continue
            columns = line.strip().split("\t")
            node_curie = columns[0]

            curie_prefix = node_curie.split(":")[0]

            # If we use different curie prefixes than the normalizer, need to fix
            normalizer_curie_prefix = curie_prefix
            normalizer_node_curie = node_curie
            if curie_prefix in self.curie_prefix_tx_arax2sri:
                normalizer_curie_prefix = self.curie_prefix_tx_arax2sri[curie_prefix]
                normalizer_node_curie = re.sub(curie_prefix, normalizer_curie_prefix, node_curie)

            # Decide if we want to keep this curie in the batch of things to look up
            # By default, no
            keep = 0
            # If this is a curie prefix that is supported by the normalizer, then yes
            if normalizer_curie_prefix in self.supported_prefixes:
                keep = 1
            # Unless it's already in the cache, then no
            if normalizer_node_curie in self.cache:
                keep = 0
            # Or if we've reached the end of the file, then set keep to 99 and trigger end-of-file processing of the last batch
            if bytes_read + 3 > filesize and len(batch) > 0:
                keep = 99

            # If we want to put this curie in the batch, or drain the batch at end-of-file
            if keep:
                if keep == 1:
                    supported_curies += 1
                    batch.append(normalizer_node_curie)

                if len(batch) > 1000 or keep == 99:
                    if bytes_read + 3 > filesize:
                        print("Drain final batch")
                    results = self.get_node_normalizer_results(batch)
                    print(".", end="", flush=True)
                    for curie in batch:
                        if curie in self.cache:
                            continue
                        curie_prefix = curie.split(":")[0]
                        if curie_prefix not in self.stats:
                            self.stats[curie_prefix] = {
                                "found": 0,
                                "not found": 0,
                                "total": 0,
                            }
                        if results is None or curie not in results or results[curie] is None:
                            self.cache[curie] = None
                            self.stats[curie_prefix]["not found"] += 1
                        else:
                            self.cache[curie] = results[curie]
                            self.stats[curie_prefix]["found"] += 1
                        self.stats[curie_prefix]["total"] += 1

                    # Clear the batch list
                    batch = []

            # Print out some progress information
            line_counter += 1
            percentage = int(bytes_read * 100.0 / filesize)
            if percentage > previous_percentage:
                previous_percentage = percentage
                print(str(percentage) + "%..", end="", flush=True)

        # Close and summarize
        fh.close()
        print("")
        print(f"{line_counter} lines read")
        print(f"{bytes_read} bytes read of {filesize} bytes in file")
        print(f"{supported_curies} curies with prefixes supported by the SRI normalizer")

        print("Build stats:")
        print(json.dumps(self.stats, indent=2, sort_keys=True))

        # Store or sync the results
        self.store_cache()

    # ############################################################################################
    # Retrieve the dict of supported BioLink types
    def get_supported_types(self):
        # If we've already done this before, return the cached result
        if self.supported_types is not None:
            return self.supported_types

        # Build the URL and fetch the result
        url = "https://nodenormalization-sri-dev.renci.org/1.1/get_semantic_types"
        response_content = requests.get(url, headers={"accept": "application/json"})
        status_code = response_content.status_code

        # Check for a returned error
        if status_code != 200:
            eprint(f"ERROR returned with status {status_code} while retrieving supported types")
            eprint(response_content)
            return

        # Unpack the response into a dict and return it
        response_dict = response_content.json()
        if "semantic_types" not in response_dict:
            eprint("ERROR Did not find expected 'semantic_types'")
            return
        if "types" not in response_dict["semantic_types"]:
            eprint("ERROR Did not find expected 'types' list")
            return

        node_types = {}
        for node_type in response_dict["semantic_types"]["types"]:
            node_types[node_type] = 1

        if len(node_types) == 0:
            node_types = None

        # Save this for future use
        self.supported_types = node_types

        return node_types

    # ############################################################################################
    # Retrieve the dict of supported curie prefixes
    def get_supported_prefixes(self):
        # If we've already done this before, return the cached result
        if self.supported_prefixes is not None:
            return self.supported_prefixes
        supported_prefixes = {}

        # Build the URL and fetch the result
        url = "https://nodenormalization-sri-dev.renci.org/1.1/get_curie_prefixes"
        response_content = requests.get(url, headers={"accept": "application/json"})
        status_code = response_content.status_code

        # Check for a returned error
        if status_code != 200:
            eprint(f"ERROR returned with status {status_code} while retrieving supported types")
            eprint(response_content)
            return

        # Unpack the response into a dict and return it
        response_dict = response_content.json()
        for entity_name, entity in response_dict.items():
            if "curie_prefix" not in entity:
                eprint("ERROR Did not find expected 'curie_prefix'")
                return
            for curie_prefix, count in entity["curie_prefix"].items():
                if str(count) != "Not found":
                    supported_prefixes[curie_prefix] = count

        # Save this for future use
        self.supported_prefixes = supported_prefixes
        return supported_prefixes

    # ############################################################################################
    # Directly fetch a normalization for a CURIE from the Normalizer
    def get_node_normalizer_results(self, curies, cache_only=None):
        if isinstance(curies, str):
            # print(f"INFO: Looking for curie {curies}")
            if curies in self.cache:
                # print(f"INFO: Using prefill cache for lookup on {curies}")
                result = {curies: self.cache[curies]}
                return result
            curies = [curies]

        if cache_only is not None:
            print(f"ERROR: Call to sri_node_normalizer requested cache_only and we missed the cache with {curies}")

        # Build the URL and fetch the result
        url = "https://nodenormalization-sri-dev.renci.org/1.1/get_normalized_nodes?"

        prefix = ""
        for curie in curies:
            url += f"{prefix}curie={curie}"
            prefix = "&"

        # eprint(f"{len(url)},")

        retry = 0
        sleep_time = 5
        error_state = True

        while retry < 3 and error_state:
            error_state = False

            try:
                response_content = requests.get(url, headers={"accept": "application/json"})
            except:
                print("Uncaught error during web request to SRI normalizer")
                error_state = True

            status_code = response_content.status_code

            # Check for a returned error
            if status_code == 404:
                # eprint(f"INFO: No normalization data for {curie}")
                return
            elif status_code != 200:
                eprint(
                    f"ERROR returned with status {status_code} while searching with URL of length {len(url)} including {curie}"
                )
                eprint(response_content)
                error_state = True

            if error_state:
                print(f"Try again after {sleep_time} seconds")
                time.sleep(sleep_time)
                retry += 1
                sleep_time += 20

        if error_state:
            return

        # Unpack the response into a dict and return it
        response_dict = response_content.json()
        return response_dict

    # ############################################################################################
    # Return a simple dict with the equivalence information and metadata about a CURIE
    def get_empty_equivalence(self, curie=""):
        response = {
            "status": "EMPTY",
            "curie": curie,
            "preferred_curie": "",
            "preferred_curie_name": "",
            "type": "",
            "equivalent_identifiers": [],
            "equivalent_names": [],
        }
        return response

    # ############################################################################################
    # Return a simple dict with the equivalence information and metadata about a CURIE
    def get_curie_equivalence(self, curie, cache_only=None):
        response = {
            "status": "ERROR",
            "curie": curie,
            "preferred_curie": "",
            "preferred_curie_name": "",
            "type": "",
            "equivalent_identifiers": [],
            "equivalent_names": [],
        }

        # Do a translation for different curie prefixes
        curie_prefix = curie.split(":")[0]
        normalizer_curie = curie
        if curie_prefix in self.curie_prefix_tx_arax2sri:
            normalizer_curie = re.sub(curie_prefix, self.curie_prefix_tx_arax2sri[curie_prefix], curie)

        results = self.get_node_normalizer_results(normalizer_curie, cache_only=cache_only)
        # print(json.dumps(results, indent=2, sort_keys=True))

        if results is None:
            response["status"] = "no information"
            return response

        # If input CURIE is not the key of the dict, this is highly unexpected
        if normalizer_curie not in results:
            eprint(f"ERROR: Did not find the curie {normalizer_curie} as a key in the results")
            return response

        if results[normalizer_curie] is None:
            response["status"] = "no information"
            return response

        # If there is no id in the results, this a highly unexpected
        if "id" not in results[normalizer_curie]:
            eprint(
                f"ERROR: Did not find 'id' as a key in the results from the SRI normalizer for curie {normalizer_curie}"
            )
            return response

        # If there is a preferred CURIE, store it and its name
        response["preferred_curie"] = results[normalizer_curie]["id"]["identifier"]
        if "label" in results[normalizer_curie]["id"]:
            response["preferred_curie_name"] = results[normalizer_curie]["id"]["label"]

        # Translate the id if necessary
        if curie != normalizer_curie:
            response["preferred_curie"] = re.sub(
                self.curie_prefix_tx_arax2sri[curie_prefix],
                curie_prefix,
                results[normalizer_curie]["id"]["identifier"],
            )

        # If there is a returned type, store it
        if "type" in results[normalizer_curie]:
            node_type = results[normalizer_curie]["type"][0]
            response["type"] = node_type

        # If there are additional equivalent identifiers and names, store them
        names = {}
        if "equivalent_identifiers" in results[normalizer_curie]:
            for equivalence in results[normalizer_curie]["equivalent_identifiers"]:
                if "identifier" in equivalence:
                    id = equivalence["identifier"]
                    if curie != normalizer_curie:
                        id = re.sub(
                            self.curie_prefix_tx_arax2sri[curie_prefix],
                            curie_prefix,
                            id,
                        )
                    # response['equivalent_identifiers'].append(id)
                    response["equivalent_identifiers"].append(equivalence)
                if "label" in equivalence:
                    if equivalence["label"] not in names:
                        response["equivalent_names"].append(equivalence["label"])

        response["status"] = "OK"
        return response


# ############################################################################################
# Command line interface for this class
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Interface to the SRI Node Normalizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--build",
        action="store_true",
        help="If set, (re)build the local SRI Node Normalizer cache from scratch",
        default=False,
    )
    parser.add_argument(
        "-l",
        "--load",
        action="store_true",
        help="If set, load the previously built local SRI Node Normalizer cache",
        default=False,
    )
    parser.add_argument(
        "-c",
        "--curie",
        action="store",
        help="Specify a curie to look up with the SRI Node Normalizer (e.g., UniProtKB:P01308, ORPHANET:2322, DRUGBANK:DB11655",
        default=None,
    )
    parser.add_argument(
        "-p",
        "--prefixes",
        action="store_true",
        help="If set, list the SRI Node Normalizer supported prefixes",
        default=None,
    )
    parser.add_argument(
        "-t",
        "--types",
        action="store_true",
        help="If set, list the SRI Node Normalizer supported types",
        default=None,
    )
    args = parser.parse_args()

    if not args.build and not args.curie and not args.prefixes and not args.types:
        parser.print_help()
        sys.exit(2)

    normalizer = SriNodeNormalizer()

    if args.prefixes:
        supported_prefixes = normalizer.get_supported_prefixes()
        print(json.dumps(supported_prefixes, indent=2, sort_keys=True))
        return

    if args.types:
        supported_types = normalizer.get_supported_types()
        print(json.dumps(supported_types, indent=2, sort_keys=True))
        return

    if args.build:
        print(
            "INFO: Beginning SRI Node Normalizer cache building process for KG2. Make sure you have a good network connection as this will download ~2 GB of data."
        )
        print(
            "INFO: Note that requests-cache is used, so the sri_node_normalizer_requests_cache.sqlite file should be deleted first if it might be stale."
        )
        normalizer.fill_cache()
        normalizer.store_cache()
        print("INFO: Build process complete")
        return

    if args.load:
        normalizer.load_cache()

    curie = "UniProtKB:P01308"
    if args.curie:
        curie = args.curie

    # print(platform.system())

    print("==========================================================")
    print("Native SRI Node Normalizer results:")
    normalized_results = normalizer.get_node_normalizer_results(curie)
    print(json.dumps(normalized_results, indent=2, sort_keys=True))

    print("==========================================================")
    print("Local more compact and useful formatting:")
    equivalence = normalizer.get_curie_equivalence(curie)
    print(json.dumps(equivalence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
