#!/usr/bin/env python3
## This code is created by RTX team (https://github.com/RTXteam/RTX/) used in the version 2.7.3 of RTX-KG2c (Wood et al. 2021 doi: 10.1101/2021.10.17.464747)
#
# Class to build and query an index of nodes in the KG
#
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


import argparse
import json
import os
import pickle
import platform
import re
import sqlite3
import timeit

from .category_manager import CategoryManager
from .sri_node_normalizer import SriNodeNormalizer

# Testing and debugging flags
DEBUG = False

DB_CONNECTION_STR = os.environ["DB_CONNECTION_STR"]


# ################################################################################################
# Main class
class NodeSynonymizer:
    # Constructor
    def __init__(self, live="Production"):
        self.options = {}
        self.kg_map = {
            "kg_nodes": {},
            "kg_unique_concepts": {},
            "kg_curies": {},
            "kg_names": {},
            "kg_name_curies": {},
        }
        self.normalizer = None
        self.exceptions = {
            "skip_SRI": {},
            "rename": {},
        }

        self.connection = None
        self.connect()

        #### Define a priority of curie prefixes. Higher is better
        self.uc_curie_prefix_scores = {
            "UMLS_STY": 5000,
            "UBERON": 3410,
            "FMA": 3400,
            "CHEMBL.COMPOUND": 3200,
            "MONDO": 2500,
            "UNIPROTKB": 2000,
            "NCBIGENE": 1900,
            "HGNC": 1850,
            "CHEMBL.TARGET": 1700,
            "DRUGBANK": 1600,
            "RXNORM": 1500,
            "VANDF": 1400,
            "CHEBI": 1100,
            "DOID": 900,
            "OMIM": 800,
            "MESH": 700,
            "UMLS": 600,
        }

    # ############################################################################################
    # Create and store a database connection
    def connect(self):
        # If already connected, don't need to do it again
        if self.connection is not None:
            return

        # Create an engine object
        if DEBUG is True:
            print("INFO: Connecting to database")

        self.connection = sqlite3.connect(DB_CONNECTION_STR)

    # ############################################################################################
    # Destroy the database connection
    def disconnect(self):
        if self.connection is None:
            if DEBUG is True:
                print("INFO: Skip disconnecting from database")
            return

        if DEBUG is True:
            print("INFO: Disconnecting from database")

        self.connection.close()
        self.connection = None

    # ############################################################################################
    # Delete and create the kgnode table
    def create_tables(self):
        print(f"INFO: Dropping and recreating tables in database {self.databaseName}")

        self.connection.execute("DROP TABLE IF EXISTS nodes")
        self.connection.execute(
            "CREATE TABLE nodes ( uc_curie VARCHAR(255), curie VARCHAR(255), original_name VARCHAR(255), adjusted_name VARCHAR(255), full_name VARCHAR(255), category VARCHAR(255), unique_concept_curie VARCHAR(255) )"
        )

        self.connection.execute("DROP TABLE IF EXISTS unique_concepts")
        self.connection.execute(
            "CREATE TABLE unique_concepts ( uc_curie VARCHAR(255), curie VARCHAR(255), name VARCHAR(255), category VARCHAR(255), normalizer_curie VARCHAR(255), normalizer_name VARCHAR(255), normalizer_category VARCHAR(255) )"
        )

        self.connection.execute("DROP TABLE IF EXISTS curies")
        self.connection.execute(
            "CREATE TABLE curies ( uc_curie VARCHAR(255), curie VARCHAR(255), unique_concept_curie VARCHAR(255), name VARCHAR(255), full_name VARCHAR(255), category VARCHAR(255), normalizer_name VARCHAR(255), normalizer_category VARCHAR(255), source VARCHAR(255) )"
        )

        self.connection.execute("DROP TABLE IF EXISTS names")
        self.connection.execute(
            "CREATE TABLE names ( lc_name VARCHAR(255), name VARCHAR(255), unique_concept_curie VARCHAR(255), source VARCHAR(255) )"
        )

        self.connection.execute("DROP TABLE IF EXISTS name_curies")
        self.connection.execute(
            "CREATE TABLE name_curies ( lc_name VARCHAR(255), name VARCHAR(255), uc_curie VARCHAR(255), unique_concept_curie VARCHAR(255), source VARCHAR(255) )"
        )

    # ############################################################################################
    def create_indexes(self):
        print("INFO: Creating INDEXes on nodes")
        self.connection.execute("CREATE INDEX idx_nodes_uc_curie ON nodes(uc_curie)")
        self.connection.execute("CREATE INDEX idx_nodes_unique_concept_curie ON nodes(unique_concept_curie)")

        print("INFO: Creating INDEXes on unique_concepts")
        self.connection.execute("CREATE INDEX idx_unique_concepts_uc_curie ON unique_concepts(uc_curie)")

        print("INFO: Creating INDEXes on curies")
        self.connection.execute("CREATE INDEX idx_curies_uc_curie ON curies(uc_curie)")
        self.connection.execute("CREATE INDEX idx_curies_unique_concept_curie ON curies(unique_concept_curie)")

        print("INFO: Creating INDEXes on names")
        self.connection.execute("CREATE INDEX idx_names_lc_name ON names(lc_name)")
        self.connection.execute("CREATE INDEX idx_names_unique_concept_curie ON names(unique_concept_curie)")

        print("INFO: Creating INDEXes on name_curies")
        self.connection.execute("CREATE INDEX idx_name_curies_lc_name ON name_curies(lc_name)")
        self.connection.execute(
            "CREATE INDEX idx_name_curies_unique_concept_curie ON name_curies(unique_concept_curie)"
        )

    # ############################################################################################
    # Import exceptions data
    def import_exceptions(self):
        #### Open the exceptions file
        with open(self.databaseLocation + "/Exceptions.txt") as infile:
            for line in infile:
                line = line.rstrip()
                if len(line) == 0:
                    continue
                if line[0] == "#":
                    continue
                is_understood = False

                match = re.match(r"skip_SRI (.+)", line)
                if match:
                    self.exceptions["skip_SRI"][match.group(1)] = True
                    print(f"INFO: Will use exception skip_SRI {match.group(1)}")
                    is_understood = True

                match = re.match(r"rename (.+?) (.+)", line)
                if match:
                    self.exceptions["rename"][match.group(1)] = match.group(2)
                    print(f"INFO: Will use exception rename {match.group(1)} = {match.group(2)}")
                    is_understood = True

                if not is_understood:
                    eprint(f"ERROR: Unable to interpret {line} in Exceptions.txt")

    # ############################################################################################
    # Create the KG node table
    def build_kg_map(self, filter_file=None):
        filename = "kg2_node_info.tsv"
        filesize = os.path.getsize(filename)

        # Correction for Windows line endings
        extra_bytes = 0
        fh = open(filename, "rb")
        if platform.system() == "Windows" and b"\r\n" in fh.read():
            print("WARNING: Windows line ending requires size compenstation")
            extra_bytes = 1
        fh.close()

        #### Import exceptions for processing
        self.import_exceptions()

        exceptions_isnot = {
            "UniProtKB:P00390": "GR",
            "UniProtKB:P04150": "GR",
            "UniProtKB:P01137": "LAP",
            "UniProtKB:P61812": "LAP",
            "UniProtKB:P10600": "LAP",
        }

        # Set up the SriNormalizer
        if self.normalizer is not None:
            normalizer = self.normalizer
        else:
            normalizer = SriNodeNormalizer()
            normalizer.load_cache()
            self.normalizer = normalizer

        normalizer_supported_categories = normalizer.get_supported_types()
        if normalizer_supported_categories is None:
            return
        normalizer_supported_prefixes = normalizer.get_supported_prefixes()
        if normalizer_supported_prefixes is None:
            return

        # The SRI NodeNormalizer conflates genes and proteins, so have a special lookup table to try to disambiguate them
        curie_prefix_categories = {
            "NCBIGene:": "biolink:Gene",
            "NCBIGENE:": "biolink:Gene",
            "ENSEMBL:ENSG": "biolink:Gene",
            "HGNC:": "biolink:Gene",
            "UniProtKB:": "biolink:Protein",
        }

        # Open the file and organize all the data
        fh = open(filename, "r", encoding="latin-1", errors="replace")
        print(f"INFO: Reading {filename} to create the NodeSynonymizer")
        lineCounter = 0
        previous_percentage = -1
        bytes_read = 0

        # Create dicts to hold all the information
        kg_nodes = self.kg_map["kg_nodes"]
        kg_unique_concepts = self.kg_map["kg_unique_concepts"]
        kg_curies = self.kg_map["kg_curies"]
        kg_names = self.kg_map["kg_names"]
        kg_name_curies = self.kg_map["kg_name_curies"]

        # For some modes of debugging, import a set of CURIEs to track
        debug_flag = DEBUG
        test_subset_flag = False
        test_set = {"identifiers": {}}
        if filter_file is not None and filter_file != False:
            print(f"INFO: Reading special testing filter_file {filter_file} for a tiny little test database")
            debug_flag = DEBUG
            test_subset_flag = True
            with open(filter_file) as debugfile:
                test_set = json.load(debugfile)
                test_set["identifiers"] = {}
                for key in test_set:
                    if key == "identifiers":
                        continue
                    for equivalence in test_set[key]["equivalent_identifiers"]:
                        id = equivalence["identifier"]
                        test_set["identifiers"][id] = 1

        # Loop over each line in the file
        for line in fh:
            #### Track the number of bytes read for progress display
            bytes_read += len(line) + extra_bytes

            #### Skip blank lines
            match = re.match(r"^\s*$", line)
            if match:
                continue

            #### Extract the columns
            columns = line.strip().split("\t")
            if len(columns) != 4:
                eprint(f"ERROR: line only has {len(columns)} at 'line'")
                exit()
            node_curie, node_name, node_full_name, node_category = columns
            uc_node_curie = node_curie.upper()
            original_node_name = node_name

            #### Skip some known problems
            if node_curie in exceptions_isnot and exceptions_isnot[node_curie] == node_name:
                continue
            if debug_flag is True and "biolink:" in node_curie:
                continue

            # Apply renaming for problem nodes
            if uc_node_curie in self.exceptions["rename"]:
                node_name = self.exceptions["rename"][uc_node_curie]
                print(
                    f"INFO: Based on manual exception, renaming {uc_node_curie} from {original_node_name} to {node_name}"
                )

            #### If we're in test subset mode, only continue the the node_curie is in the test subset
            if test_subset_flag:
                if node_curie not in test_set["identifiers"]:
                    continue

            #### For debugging problems
            # debug_flag = False
            # if 'HGNC:29603' in node_curie: debug_flag = True
            # if node_name.lower() == 'ache': debug_flag = True

            if debug_flag:
                print("===============================================")
                print(f"Input: {line.strip()}")

            # Perform some data scrubbing
            scrubbed_values = self.scrub_input(node_curie, node_name, node_category, debug_flag)
            node_curie = scrubbed_values["node_curie"]
            uc_node_curie = node_curie.upper()
            node_name = scrubbed_values["node_name"]
            node_category = scrubbed_values["node_category"]
            names = scrubbed_values["names"]
            curie_prefix = node_curie.split(":")[0]
            if debug_flag:
                print("Final name list: ", names)

            # See if this curie is already a curie we've encountered before, figure out its unique_concept_curie
            unique_concept_curie = None
            equivalence = None
            if uc_node_curie in kg_curies:
                unique_concept_curie = kg_curies[uc_node_curie]["curie"]
                uc_unique_concept_curie = kg_curies[uc_node_curie]["uc_unique_concept_curie"]
                if debug_flag:
                    print(
                        f"DEBUG: This curie was already seen. Setting to its uc_unique_concept_curie to {uc_unique_concept_curie}"
                    )

            # Check to see if this is a supported prefix or in the translation table
            if curie_prefix in normalizer.curie_prefix_tx_arax2sri or curie_prefix in normalizer_supported_prefixes:
                if node_curie in self.exceptions["skip_SRI"]:
                    print(f"WARNING: Skipping SRI NN lookup for {node_curie} due to directive in Exceptions.txt file")
                    equivalence = {
                        "status": "SRI NN skipped per exceptions",
                        "equivalent_identifiers": [],
                        "equivalent_names": [],
                        "preferred_curie": "",
                        "preferred_curie_name": "",
                        "type": "",
                    }

                else:
                    equivalence = normalizer.get_curie_equivalence(node_curie, cache_only=True)
                    if debug_flag:
                        print(
                            "DEBUG: SRI normalizer returned: ",
                            json.dumps(equivalence, indent=2, sort_keys=True),
                        )

                # Apply renaming for problem nodes
                uc_preferred_curie = equivalence["preferred_curie"].upper()
                if uc_preferred_curie in self.exceptions["rename"]:
                    print(
                        f"INFO: Based on manual exception, renaming SRI node normalizer result {uc_preferred_curie} from {equivalence['preferred_curie_name']} to {self.exceptions['rename'][uc_preferred_curie]}"
                    )
                    equivalence["preferred_curie_name"] = self.exceptions["rename"][uc_preferred_curie]
                    equivalence["equivalent_names"] = [self.exceptions["rename"][uc_preferred_curie]]

                # Extract the preferred designation of the normalizer
                normalizer_curie = equivalence["preferred_curie"]
                normalizer_name = equivalence["preferred_curie_name"]
                normalizer_category = equivalence["type"]

            # Else just warn that there's nothing to look for
            else:
                if debug_flag:
                    print(f"WARNING: CURIE prefix '{curie_prefix}' not supported by normalizer. Skipped.")
                equivalence = {
                    "status": "category not supported",
                    "equivalent_identifiers": [],
                    "equivalent_names": [],
                }
                normalizer_curie = ""
                normalizer_name = ""
                normalizer_category = ""

            # If the normalizer has something for us, then use that as the unique_concept_curie
            overridden_normalizer_category = normalizer_category
            if equivalence["status"] == "OK":
                # Unless the normalizer category is a gene and the current category is a protein. Then keep it a protein because we are protein-centric
                if normalizer_category == "biolink:Gene" and node_category == "biolink:Protein":
                    if debug_flag:
                        print(
                            f"DEBUG: Since this is protein and the normalizer says gene, stay with this one as the unique_concept {node_curie}"
                        )
                    unique_concept_curie = node_curie
                    uc_unique_concept_curie = node_curie.upper()
                    overridden_normalizer_category = node_category
                else:
                    if debug_flag:
                        print(f"DEBUG: Using the SRI normalizer normalized unique_concept {normalizer_curie}")
                    unique_concept_curie = normalizer_curie
                    uc_unique_concept_curie = normalizer_curie.upper()

            # And if the normalizer did not have anything for us
            else:
                # If we've already seen this synonym, then switch to that unique concept
                # I hope this will save RAM by not creating so many unique concepts that must be coalesced later
                if node_name.lower() in kg_names:
                    uc_unique_concept_curie = kg_names[node_name.lower()]["uc_unique_concept_curie"]
                    unique_concept_curie = kg_unique_concepts[uc_unique_concept_curie]["curie"]
                    if debug_flag:
                        print(f"DEBUG: Found an existing unique concept by synonym as {unique_concept_curie}")

                # Else this node becomes the unique_concept_curie
                else:
                    if debug_flag:
                        print("DEBUG: This node will become its own unique concept")
                    unique_concept_curie = node_curie
                    uc_unique_concept_curie = node_curie.upper()

            # Place this curie in the index
            kg_curies[uc_node_curie] = {
                "curie": node_curie,
                "uc_unique_concept_curie": uc_unique_concept_curie,
                "name": node_name,
                "full_name": node_full_name,
                "category": node_category,
                "normalizer_name": normalizer_name,
                "normalizer_category": normalizer_category,
                "source": "KG2",
            }

            # If this unique_concept_curie already exists, embrace it
            if uc_unique_concept_curie in kg_unique_concepts:
                # Put the current curie in there. Names will be done below.
                kg_unique_concepts[uc_unique_concept_curie]["all_uc_curies"][uc_node_curie] = 1

            # Otherwise, create the entry
            else:
                # If we got something from the SRI normalizer, start with that
                if equivalence["status"] == "OK":
                    if debug_flag:
                        print("DEBUG: Create new unique concept based on SRI normalizer info")
                    kg_unique_concepts[uc_unique_concept_curie] = {
                        "curie": unique_concept_curie,
                        "name": normalizer_name,
                        "category": overridden_normalizer_category,
                        "normalizer_curie": normalizer_curie,
                        "normalizer_name": normalizer_name,
                        "normalizer_category": normalizer_category,
                        "all_uc_curies": {uc_node_curie: True},
                        "all_lc_names": {node_name.lower(): True},
                    }

                # Otherwise, we use this node
                else:
                    if debug_flag:
                        print("DEBUG: Create new unique concept based on this node")
                    kg_unique_concepts[uc_unique_concept_curie] = {
                        "curie": unique_concept_curie,
                        "name": node_name,
                        "category": node_category,
                        "normalizer_curie": None,
                        "normalizer_name": None,
                        "normalizer_category": None,
                        "all_uc_curies": {uc_node_curie: True},
                        "all_lc_names": {node_name.lower(): True},
                    }

            # Loop through the equivalent identifiers from the SRI normalizer and add those to the list
            for equivalent_concept in equivalence["equivalent_identifiers"]:
                equivalent_identifier = equivalent_concept["identifier"]
                equivalent_name = None
                if "label" in equivalent_concept:
                    equivalent_name = equivalent_concept["label"]

                # Try to deconflate gene and protein
                this_category = normalizer_category
                if this_category == "":
                    this_category = node_category
                for curie_prefix_category in curie_prefix_categories:
                    if equivalent_identifier.startswith(curie_prefix_category):
                        this_category = curie_prefix_categories[curie_prefix_category]
                uc_equivalent_identifier = equivalent_identifier.upper()

                # If this equivalient identifier is already there (either from KG2 or previous SRI NN encounter), store the normalizer information
                if uc_equivalent_identifier in kg_curies:
                    kg_curies[uc_equivalent_identifier]["normalizer_name"] = equivalent_name
                    kg_curies[uc_equivalent_identifier]["normalizer_category"] = this_category
                    if "SRI_NN" not in kg_curies[uc_equivalent_identifier]["source"]:
                        kg_curies[uc_equivalent_identifier]["source"] += ",SRI_NN"

                    # Turns out this is not always true as one would have expected. One example is HSFX1 and HSFX2 with
                    # NCBIGene:100130086 and NCBIGene:100506164 but yet they are tied by one protein UniProtKB:Q9UBD0
                    # This system will coalesce them, although the normalizer has them separate
                    if 0 and uc_unique_concept_curie != kg_curies[uc_equivalent_identifier]["uc_unique_concept_curie"]:
                        print(
                            f"ERROR 247: at node_curie={node_curie}, expected {uc_unique_concept_curie} == {kg_curies[uc_equivalent_identifier]['uc_unique_concept_curie']}, but no."
                        )
                        if debug_flag:
                            print(
                                f"kg_curies[{uc_equivalent_identifier}] = "
                                + json.dumps(
                                    kg_curies[uc_equivalent_identifier],
                                    indent=2,
                                    sort_keys=True,
                                )
                            )
                            print(
                                f"kg_unique_concepts[{uc_equivalent_identifier}] = "
                                + json.dumps(
                                    kg_unique_concepts[uc_equivalent_identifier],
                                    indent=2,
                                    sort_keys=True,
                                )
                            )
                            sys.exit(1)

                # If we haven't seen this CURIE before, and we first hear about from the SRI NN, then create are record for it
                else:
                    kg_curies[uc_equivalent_identifier] = {
                        "curie": equivalent_identifier,
                        "uc_unique_concept_curie": uc_unique_concept_curie,
                        "name": None,
                        "full_name": None,
                        "category": None,
                        "normalizer_name": equivalent_name,
                        "normalizer_category": this_category,
                        "source": "SRI_NN",
                    }
                    kg_unique_concepts[uc_unique_concept_curie]["all_uc_curies"][uc_equivalent_identifier] = True

            # Loop through the equivalent names from the SRI normalizer and add those to the list
            for equivalent_name in equivalence["equivalent_names"]:
                if equivalent_name == "":
                    print(f"ERROR: SRI normalizer equivalent_name is '' for {uc_unique_concept_curie}")
                    exit()

                # If this equivalent name is already there, just make sure the unique_concept_curie is the same
                lc_equivalent_name = equivalent_name.lower()
                if lc_equivalent_name in kg_names:
                    # print(f"INFO: Adding {unique_concept_curie} to synonym {lc_equivalent_name}")
                    kg_names[lc_equivalent_name]["uc_unique_concept_curies"][uc_unique_concept_curie] = 1
                # If not, then create it
                else:
                    kg_names[lc_equivalent_name] = {
                        "name": equivalent_name,
                        "uc_unique_concept_curie": uc_unique_concept_curie,
                        "source": "SRI",
                        "uc_unique_concept_curies": {uc_unique_concept_curie: 1},
                    }
                    kg_unique_concepts[uc_unique_concept_curie]["all_lc_names"][lc_equivalent_name] = (
                        1  # FIXME. A count would be fun
                    )

            # If there is already a kg_nodes entry for this curie, then assume this is a synonym
            if uc_node_curie in kg_nodes:
                uc_unique_concept_curie = kg_nodes[uc_node_curie]["uc_unique_concept_curie"]

                # For now, just ignore cases where the categories are different. It happens alot
                # if node_category != kg_nodes[uc_node_curie]['category']:
                #    print(f"ERROR 249: at node_curie={node_curie}, expected {node_category} == {kg_nodes[uc_node_curie]['category']}, but no.")

                # Update the KG presence. If the current value is not the current KG, then it much be both
                # if kg_nodes[uc_node_curie]['kg_presence'] != kg_name:
                #    kg_nodes[uc_node_curie]['kg_presence'] = 'KG1,KG2'

            # Otherwise, create the entry
            else:
                kg_nodes[uc_node_curie] = {
                    "curie": node_curie,
                    "original_name": original_node_name,
                    "adjusted_name": node_name,
                    "full_name": node_full_name,
                    "category": node_category,
                    "uc_unique_concept_curie": uc_unique_concept_curie,
                }

            # Loop over all scrubbed names for this node to insert kg_names
            for equivalent_name in names:
                lc_equivalent_name = equivalent_name.lower()

                #### If the name is empty or otherwise blank, then we will not add it to kg_names
                if equivalent_name is None or equivalent_name == "":
                    # print(f"INFO: Will not record blank name for {node_curie} in kg_names")
                    continue
                match = re.match(r"^\s+$", equivalent_name)
                if match:
                    print(f"WARNING: equivalent_name for {node_curie} is whitespace but not empty ({equivalent_name})")
                    continue

                # If this equivalent name is already there, just make sure the unique_concept_curie is the same
                if lc_equivalent_name in kg_names:
                    if debug_flag:
                        print(f"DEBUG: Name '{lc_equivalent_name}' already in kg_names")
                    if uc_unique_concept_curie != kg_names[lc_equivalent_name]["uc_unique_concept_curie"]:
                        kg_names[lc_equivalent_name]["uc_unique_concept_curies"][uc_unique_concept_curie] = 1
                        if debug_flag:
                            print(
                                f"INFO: uc_unique_concept_curie={uc_unique_concept_curie}, but kg_names already has {kg_names[lc_equivalent_name]['uc_unique_concept_curie']}. Oh well, this will be cleaned up later."
                            )
                            print(f"INFO: * Adding {uc_unique_concept_curie} to kg_names {lc_equivalent_name}")
                            print(
                                f"INFO: **** {lc_equivalent_name} has {kg_names[lc_equivalent_name]['uc_unique_concept_curies']}"
                            )

                # If not, then create it
                else:
                    if debug_flag:
                        print(f"DEBUG: Name '{lc_equivalent_name}' is not in kg_names. Add it")
                        print(
                            f"       node_curie={node_curie}, uc_node_curie={uc_node_curie}, uc_unique_concept_curie={uc_unique_concept_curie}, lc_equivalent_name={lc_equivalent_name}"
                        )
                    kg_names[lc_equivalent_name] = {
                        "name": equivalent_name,
                        "uc_unique_concept_curie": uc_unique_concept_curie,
                        "source": "KG2",
                        "uc_unique_concept_curies": {uc_unique_concept_curie: 1},
                    }
                    kg_unique_concepts[uc_unique_concept_curie]["all_lc_names"][lc_equivalent_name] = 1

            # Debugging
            if debug_flag:
                print(
                    f"kg_nodes['{uc_node_curie}'] = ",
                    json.dumps(kg_nodes[uc_node_curie], indent=2, sort_keys=True),
                )
                print(
                    f"kg_unique_concepts['{uc_unique_concept_curie}'] = ",
                    json.dumps(
                        kg_unique_concepts[uc_unique_concept_curie],
                        indent=2,
                        sort_keys=True,
                    ),
                )
                # input("Enter to continue...")

            # debug_flag = False
            lineCounter += 1
            percentage = int(bytes_read * 100.0 / filesize)
            if percentage > previous_percentage:
                previous_percentage = percentage
                print(str(percentage) + "%..", end="", flush=True)

        fh.close()
        print("")

        print("INFO: Freeing SRI node normalizer cache from memory")
        self.normalizer = None
        del normalizer

        print("INFO: Reading of KG2 node files complete")
        print(
            f"INFO: Current entity counts\n  {len(kg_nodes)} nodes\n  {len(kg_unique_concepts)} unique concepts\n"
            + f"  {len(kg_curies)} curies\n  {len(kg_names)} names and abbreviations\n"
            + f"  {len(kg_name_curies)} name to curie provenance associations"
        )

    # ############################################################################################
    def save_state(self):
        filename = "node_synonymizer_map_state.pickle"
        print(f"INFO: Writing the state to {filename}")

        try:
            with open(filename, "wb") as outfile:
                pickle.dump(self.kg_map, outfile)
            return True
        except:
            print(f"ERROR: Unable to save state to {filename}")
            return None

    # ############################################################################################
    def reload_state(self):
        filename = "node_synonymizer_map_state.pickle"
        print(f"INFO: Loading previous data structure state from {filename}")

        try:
            with open(filename, "rb") as infile:
                self.kg_map = pickle.load(infile)
            return True
        except:
            print(f"ERROR: Unable to reload previous state from {filename}")
            return None

        print(
            f"INFO: Finished loading previous data structure state from {filename}. Have {len(self.kg_map[kg_nodes])} kg_nodes."
        )

    # ############################################################################################
    # def show_state(self, concept='insulin'):
    #    uc_unique_concept_curie = kg_names[concept]['uc_unique_concept_curie']
    #    uc_node_curie = uc_unique_concept_curie
    #    if kg_unique_concepts[uc_unique_concept_curie]['remapped_curie'] is not None:
    #        uc_node_curie = kg_unique_concepts[uc_unique_concept_curie]['remapped_curie']
    #    print(f"kg_nodes['{uc_node_curie}'] = ",json.dumps(kg_nodes[uc_node_curie], indent=2, sort_keys=True))
    #    print(f"kg_unique_concepts['{uc_unique_concept_curie}'] = ",json.dumps(kg_unique_concepts[uc_unique_concept_curie], indent=2, sort_keys=True))

    # ############################################################################################
    #### Store the built-up in-memory index to the database
    def store_kg_map(self):
        kg_nodes = self.kg_map["kg_nodes"]
        kg_unique_concepts = self.kg_map["kg_unique_concepts"]
        kg_curies = self.kg_map["kg_curies"]
        kg_names = self.kg_map["kg_names"]
        kg_name_curies = self.kg_map["kg_name_curies"]

        # Write all nodes
        n_rows = len(kg_nodes)
        i_rows = 0
        previous_percentage = -1
        rows = []
        print(f"INFO: Writing {n_rows} nodes to the database")
        for uc_curie, node in kg_nodes.items():
            rows.append(
                [
                    uc_curie,
                    node["curie"],
                    node["original_name"],
                    node["adjusted_name"],
                    node["full_name"],
                    node["category"],
                    node["uc_unique_concept_curie"],
                ]
            )
            i_rows += 1
            if i_rows == int(i_rows / 5000.0) * 5000 or i_rows == n_rows:
                self.connection.executemany(
                    "INSERT INTO nodes (uc_curie, curie, original_name, adjusted_name, full_name, category, unique_concept_curie) values (?,?,?,?,?,?,?)",
                    rows,
                )
                self.connection.commit()
                rows = []
                percentage = int(i_rows * 100.0 / n_rows)
                if percentage > previous_percentage:
                    previous_percentage = percentage
                    print(str(percentage) + "%..", end="", flush=True)

        # Write all unique concepts
        n_rows = len(kg_unique_concepts)
        i_rows = 0
        previous_percentage = -1
        rows = []
        print(f"\nINFO: Writing {n_rows} unique_concepts to the database")
        for uc_curie, concept in kg_unique_concepts.items():
            rows.append(
                [
                    uc_curie,
                    concept["curie"],
                    concept["name"],
                    concept["category"],
                    concept["normalizer_curie"],
                    concept["normalizer_name"],
                    concept["normalizer_category"],
                ]
            )
            i_rows += 1
            if i_rows == int(i_rows / 5000.0) * 5000 or i_rows == n_rows:
                self.connection.executemany(
                    "INSERT INTO unique_concepts (uc_curie, curie, name, category, normalizer_curie, normalizer_name, normalizer_category) values (?,?,?,?,?,?,?)",
                    rows,
                )
                self.connection.commit()
                rows = []
                percentage = int(i_rows * 100.0 / n_rows)
                if percentage > previous_percentage:
                    previous_percentage = percentage
                    print(str(percentage) + "%..", end="", flush=True)

        # Write all curies
        n_rows = len(kg_curies)
        i_rows = 0
        previous_percentage = -1
        rows = []
        print(f"\nINFO: Writing {n_rows} curies to the database")
        for uc_curie, concept in kg_curies.items():
            rows.append(
                [
                    uc_curie,
                    concept["curie"],
                    concept["uc_unique_concept_curie"],
                    concept["name"],
                    concept["full_name"],
                    concept["category"],
                    concept["normalizer_name"],
                    concept["normalizer_category"],
                    concept["source"],
                ]
            )
            i_rows += 1
            if i_rows == int(i_rows / 5000.0) * 5000 or i_rows == n_rows:
                self.connection.executemany(
                    "INSERT INTO curies (uc_curie, curie, unique_concept_curie, name, full_name, category, normalizer_name, normalizer_category, source) values (?,?,?,?,?,?,?,?,?)",
                    rows,
                )
                self.connection.commit()
                rows = []
                percentage = int(i_rows * 100.0 / n_rows)
                if percentage > previous_percentage:
                    previous_percentage = percentage
                    print(str(percentage) + "%..", end="", flush=True)

        # Write all synonyms
        n_rows = len(kg_names)
        i_rows = 0
        previous_percentage = -1
        rows = []
        print(f"\nINFO: Writing {n_rows} names to the database")
        for lc_synonym_name, synonym in kg_names.items():
            rows.append(
                [
                    lc_synonym_name,
                    synonym["name"],
                    synonym["uc_unique_concept_curie"],
                    synonym["source"],
                ]
            )
            i_rows += 1
            if i_rows == int(i_rows / 5000.0) * 5000 or i_rows == n_rows:
                self.connection.executemany(
                    "INSERT INTO names (lc_name, name, unique_concept_curie, source) values (?,?,?,?)",
                    rows,
                )
                self.connection.commit()
                rows = []
                percentage = int(i_rows * 100.0 / n_rows)
                if percentage > previous_percentage:
                    previous_percentage = percentage
                    print(str(percentage) + "%..", end="", flush=True)
        print("")

        # Write all name-curies
        n_rows = len(kg_name_curies)
        i_rows = 0
        previous_percentage = -1
        rows = []
        print(f"\nINFO: Writing {n_rows} name_curies to the database")
        for lc_name_uc_curie, synonym in kg_name_curies.items():
            rows.append(
                [
                    synonym["name"].lower(),
                    synonym["name"],
                    synonym["uc_curie"],
                    synonym["uc_unique_concept_curie"],
                    synonym["source"],
                ]
            )
            i_rows += 1
            if i_rows == int(i_rows / 5000.0) * 5000 or i_rows == n_rows:
                self.connection.executemany(
                    "INSERT INTO name_curies (lc_name, name, uc_curie, unique_concept_curie, source) values (?,?,?,?,?)",
                    rows,
                )
                self.connection.commit()
                rows = []
                percentage = int(i_rows * 100.0 / n_rows)
                if percentage > previous_percentage:
                    previous_percentage = percentage
                    print(str(percentage) + "%..", end="", flush=True)
        print("")

    #############################################################################################
    #### Go through all unique concepts and merge any that are split due to build order issues
    def merge_unique_concepts(self):
        print("INFO: Merging unique concepts by existing uc_curie associations...")

        kg_nodes = self.kg_map["kg_nodes"]
        kg_unique_concepts = self.kg_map["kg_unique_concepts"]
        kg_curies = self.kg_map["kg_curies"]
        kg_names = self.kg_map["kg_names"]
        kg_name_curies = self.kg_map["kg_name_curies"]

        all_uc_curies_to_unique_concepts = {}

        debug_flag = DEBUG

        #### Progress tracking
        counter = 0
        n_items = len(kg_unique_concepts)
        previous_percentage = 0

        #### Loop over all unique_concepts to see if they need coalescing
        print("INFO: Looping over all unique_concepts...")
        for uc_curie, concept in kg_unique_concepts.items():
            if debug_flag:
                print("====================================================================")
                print(
                    f"==== {uc_curie}  {concept['name']}  {concept['category']} has {len(concept['all_uc_curies'])} items"
                )
                print(json.dumps(concept, indent=2, sort_keys=True))

            #### Loop over all the uc_curies that map to this concept, recording the mapping in all_uc_curies_to_unique_concepts
            #### and scoring the unique_concept curie prefixes
            for uc_equivalent_curie in concept["all_uc_curies"]:
                #### If this curie hasn't been encountered yet, add an entry to the dict
                if uc_equivalent_curie not in all_uc_curies_to_unique_concepts:
                    all_uc_curies_to_unique_concepts[uc_equivalent_curie] = {}

                #### Compute a score for this uc_unique_concept_curie
                score = 0
                # uc_curie_prefix = uc_curie.split(':')[0]
                uc_curie_prefix = uc_equivalent_curie.split(":")[0]
                if uc_curie_prefix in self.uc_curie_prefix_scores:
                    score = self.uc_curie_prefix_scores[uc_curie_prefix]
                if uc_curie_prefix == "UNIPROTKB":
                    if ":P" in uc_curie:
                        score += 2
                    if ":Q" in uc_curie:
                        score += 1

                #### Store the mapping and score
                all_uc_curies_to_unique_concepts[uc_equivalent_curie][uc_curie] = {
                    "category": kg_unique_concepts[uc_curie]["category"],
                    "score": score,
                }

                if debug_flag:
                    print(f"{uc_equivalent_curie} -> {uc_curie} = {score}")

            counter += 1
            percentage = int(counter * 100.0 / n_items)
            if percentage > previous_percentage:
                previous_percentage = percentage
                print(str(percentage) + "%..", end="", flush=True)

        print(
            f"INFO: After scanning {n_items} distinct unique_concepts, all_uc_curies_to_unique_concepts has {len(all_uc_curies_to_unique_concepts)} items"
        )

        #### Loop through all the mappings to find and prioritize the sets
        print("INFO: Looping over all all_uc_curies_to_unique_concepts...")
        concept_remap = {}
        for uc_unique_concept_curie, group in all_uc_curies_to_unique_concepts.items():
            if len(group) > 1:
                if debug_flag:
                    print(f"Assessing {uc_unique_concept_curie}")
                    print(f"Need to resolve a group {group}")
                best_score = -1
                best_curie = uc_unique_concept_curie

                #### Determine the best scoring one (also sorted in case of ties)
                for member_curie in sorted(group):
                    member = group[member_curie]
                    if member["score"] > best_score:
                        best_score = member["score"]
                        best_curie = member_curie

                #### Remap all the others to the best scoring one
                for member_curie in group:
                    if member_curie != best_curie:
                        concept_remap[member_curie] = best_curie
                        if debug_flag:
                            print(f"  ** {member_curie} -> {best_curie}")
        #### Perform the remapping
        self.remap_concepts(concept_remap)

    #############################################################################################
    #### Given a concept_remap dict, go ahead and perform all the remapping
    def remap_concepts(self, concept_remap):
        print("INFO: Remapping concepts using concept translation map...")

        kg_nodes = self.kg_map["kg_nodes"]
        kg_unique_concepts = self.kg_map["kg_unique_concepts"]
        kg_curies = self.kg_map["kg_curies"]
        kg_names = self.kg_map["kg_names"]
        kg_name_curies = self.kg_map["kg_name_curies"]

        #### At this point, there is remapping information, but it's possible to have
        #### A->B, B->C and we want, A->C, B->C
        #### So we need to iterate through and look for targets that are also sources
        iteration = 0
        n_prev_forward_entries = -1
        while True:
            new_concept_remap = {}
            forwarded_entries = {}
            is_new_map_changed = False
            iteration += 1
            for uc_unique_concept_curie, target_curie in concept_remap.items():
                if target_curie in concept_remap and target_curie != concept_remap[target_curie]:
                    forwarded_entries[target_curie] = concept_remap[target_curie]
                    target_curie = concept_remap[target_curie]
                    is_new_map_changed = True
                new_concept_remap[uc_unique_concept_curie] = target_curie
            if is_new_map_changed:
                concept_remap = new_concept_remap
                print(f"- After iteration {iteration}, there are {len(forwarded_entries)} forwarded entries")
                if len(forwarded_entries) < 10:
                    print(json.dumps(forwarded_entries, sort_keys=True, indent=2))
            else:
                break
            if len(forwarded_entries) == n_prev_forward_entries:
                print(f"- Stopped iterating. Two iterations with {len(forwarded_entries)} forwarded entries")
                break
            if iteration > 123:
                eprint("ERROR: E98123: Reached 123 iterations")
                exit()
            n_prev_forward_entries = len(forwarded_entries)

        #### Show the mapping
        for uc_unique_concept_curie, target_curie in concept_remap.items():
            # print(f"{uc_unique_concept_curie} --> {target_curie}")
            # Transfer the all_uc_curies entries
            for uc_curie in kg_unique_concepts[uc_unique_concept_curie]["all_uc_curies"]:
                if uc_curie not in kg_unique_concepts[target_curie]["all_uc_curies"]:
                    kg_unique_concepts[target_curie]["all_uc_curies"][uc_curie] = True
                if DEBUG:
                    print(
                        f"INFO: Transferring {uc_curie} ({kg_curies[uc_curie]['name']}) from all_uc_curies in {uc_unique_concept_curie} ({kg_unique_concepts[uc_unique_concept_curie]['name']}) to {target_curie} ({kg_unique_concepts[target_curie]['name']})"
                    )

        #### Remap kg_nodes
        for uc_curie, element in kg_nodes.items():
            if element["uc_unique_concept_curie"] in concept_remap:
                element["uc_unique_concept_curie"] = concept_remap[element["uc_unique_concept_curie"]]

        #### Remap kg_curies
        for uc_curie, element in kg_curies.items():
            if element["uc_unique_concept_curie"] in concept_remap:
                element["uc_unique_concept_curie"] = concept_remap[element["uc_unique_concept_curie"]]

        #### Remap kg_names
        for lc_name, element in kg_names.items():
            if element["uc_unique_concept_curie"] in concept_remap:
                #### Reassign the uc_unique_concept_curie
                element["uc_unique_concept_curie"] = concept_remap[element["uc_unique_concept_curie"]]

            #### Add in the new one
            element["uc_unique_concept_curies"][element["uc_unique_concept_curie"]] = True

            #### Remove any remapped uc_curies
            new_uc_unique_concept_curies = {}
            for uc_curie in element["uc_unique_concept_curies"]:
                if uc_curie not in new_concept_remap:
                    new_uc_unique_concept_curies[uc_curie] = True
            element["uc_unique_concept_curies"] = new_uc_unique_concept_curies

        #### Remap kg_name_curies
        for lc_name_uc_curie, element in kg_name_curies.items():
            if element["uc_unique_concept_curie"] in concept_remap:
                element["uc_unique_concept_curie"] = concept_remap[element["uc_unique_concept_curie"]]

        #### And delete the obsolete unique_concepts
        for uc_curie in concept_remap:
            del kg_unique_concepts[uc_curie]

        print("INFO: Done remapping")
        print(
            f"INFO: Current entity counts\n  {len(kg_nodes)} nodes\n  {len(kg_unique_concepts)} unique concepts\n"
            + f"  {len(kg_curies)} curies\n  {len(kg_names)} names and abbreviations\n"
            + f"  {len(kg_name_curies)} name to curie provenance associations"
        )

    #############################################################################################
    #### Go through all unique concepts and merge any that are split due to build order issues
    def merge_unique_concepts_by_name(self):
        print("INFO: Merging unique concepts by name...")

        kg_nodes = self.kg_map["kg_nodes"]
        kg_unique_concepts = self.kg_map["kg_unique_concepts"]
        kg_curies = self.kg_map["kg_curies"]
        kg_names = self.kg_map["kg_names"]

        all_lc_names_to_unique_concepts = {}
        n_concepts_to_merge = 0

        debug_flag = DEBUG

        #### Progress tracking
        counter = 0
        n_items = len(kg_names)
        previous_percentage = 0

        # Loop over all names
        print("INFO: Looping over all names...")
        for lc_name, concept in kg_names.items():
            # if lc_name == 'losartan':
            #    debug_flag = True

            if len(lc_name) < 3:
                continue

            if debug_flag:
                print("====================================================================")
                print(f"==== {lc_name}  {concept['name']}  {concept['uc_unique_concept_curie']}")
                print(json.dumps(concept, indent=2, sort_keys=True))

            # Loop over all the equivalences, picking the best ones
            for uc_curie in concept["uc_unique_concept_curies"]:
                #### If this curie hasn't been seen yet, then add it to the dict
                if lc_name not in all_lc_names_to_unique_concepts:
                    all_lc_names_to_unique_concepts[lc_name] = {}

                #### Compute a score for this uc_unique_concept_curie
                score = 0
                uc_curie_prefix = uc_curie.split(":")[0]
                if uc_curie_prefix in self.uc_curie_prefix_scores:
                    score = self.uc_curie_prefix_scores[uc_curie_prefix]
                if uc_curie_prefix == "UNIPROTKB":
                    if ":P" in uc_curie:
                        score += 2
                    if ":Q" in uc_curie:
                        score += 1

                #### Store the mapping and score
                all_lc_names_to_unique_concepts[lc_name][uc_curie] = {
                    "category": kg_unique_concepts[uc_curie]["category"],
                    "score": score,
                }

                # debug_flag = False
                if debug_flag:
                    print(f"{lc_name} -> {uc_curie} = {score}")

            counter += 1
            percentage = int(counter * 100.0 / n_items)
            if percentage > previous_percentage:
                previous_percentage = percentage
                print(str(percentage) + "%..", end="", flush=True)

        print(
            f"INFO: After scanning {n_items} distinct names, all_lc_names_to_unique_concepts has {len(all_lc_names_to_unique_concepts)} items"
        )

        #### Loop through all the mappings to find and prioritize the sets
        print("INFO: Looping over all all_lc_names_to_unique_concepts...")
        concept_remap = {}
        for lc_name, group in all_lc_names_to_unique_concepts.items():
            if len(group) > 1:
                if debug_flag:
                    print(f"Need to resolve a group {group}")
                best_score = -1
                best_curie = ""

                #### Determine the best scoring one (also sorted in case of ties)
                for member_curie in sorted(group):
                    member = group[member_curie]
                    if member["score"] > best_score:
                        best_score = member["score"]
                        best_curie = member_curie

                #### Remap all the others to the best scoring one
                for member_curie in group:
                    if member_curie != best_curie:
                        concept_remap[member_curie] = best_curie

        #### Perform the remapping
        self.remap_concepts(concept_remap)

    # ############################################################################################
    #### The input lines are a bit messy. Here is special code to tidy things up a bit using hand curated heuristics
    def scrub_input(self, node_curie, node_name, node_category, debug_flag):
        # Many MONDO names have a ' (disease)' suffix, which seems undesirable, so strip them out
        if "MONDO:" in node_curie:
            node_name = re.sub(r"\s*\(disease\)\s*$", "", node_name)
        # Many PR names have a ' (human)' suffix, which seems undesirable, so strip them out
        if "PR:" in node_curie or "HGNC:" in node_curie:
            node_name = re.sub(r"\s*\(human\)\s*$", "", node_name)
        # Many ENSEMBLs have  [Source:HGNC Symbol;Acc:HGNC:29884], which seems undesirable, so strip them out
        if "ENSEMBL:" in node_curie:
            node_name = re.sub(r"\s*\[Source:HGNC.+\]\s*", "", node_name)

        # Create a list of all the possible names we will add to the database
        names = {node_name: 0}

        # OMIM often has multiple names separated by semi-colon. Separate them
        if re.match("OMIM:", node_curie):
            multipleNames = node_name.split("; ")
            if len(multipleNames) > 1:
                for possibleName in multipleNames:
                    #### Changed behavior 2020-12-14 to only keep the first in the list of semi-colon separated, so as to avoid gene symbol clashes, see #1165
                    if possibleName == multipleNames[0]:
                        # next
                        break
                    names[possibleName] = 0
                    break

        # Reactome names sometimes have an abbrevation in parentheses. Extract it and store both the abbreviation and the name without it
        elif re.match("REACT:R-HSA-", node_curie):
            # Also store the path name without embedded abbreviations
            match = re.search(r" \(([A-Z0-9\-]{1,8})\)", node_name)
            if match:
                names[match.group(1)] = 0
                newName = re.sub(r" \([A-Z0-9\-]{1,8}\)", "", node_name, flags=re.IGNORECASE)
                names[newName] = 0

        # If this is a UniProt identifier, add a synonym that is the naked identifier without the prefix
        elif re.match("UniProtKB:[A-Z][A-Z0-9]{5}", node_curie) or re.match("UniProtKB:A[A-Z0-9]{9}", node_curie):
            tmp = re.sub("UniProtKB:", "", node_curie)
            names[tmp] = 0

        # If this is a PR identifier, add a synonym that is the naked identifier without the prefix
        elif re.match("PR:[A-Z][A-Z0-9]{5}", node_curie) or re.match("PR:A[A-Z0-9]{9}", node_curie):
            tmp = re.sub("PR:", "", node_curie)
            names[tmp] = 0

        # Create duplicates for various DoctorName's diseases
        # Cannot add to names{} in place while looping, so put in a new dict and then append when done
        more_names = {}
        for name in names:
            if re.search("'s ", name):
                newName = re.sub("'s ", "s ", name)
                more_names[newName] = 0
                newName = re.sub("'s ", " ", name)
                more_names[newName] = 0
        for name in more_names:
            names[name] = more_names[name]

        # A few special cases
        if re.search("alzheimer ", node_name, flags=re.IGNORECASE):
            newName = re.sub("alzheimer ", "alzheimers ", node_name, flags=re.IGNORECASE)
            names[newName] = 0

            newName = re.sub("alzheimer ", "alzheimer's ", node_name, flags=re.IGNORECASE)
            names[newName] = 0

        # Define a set of names that we will supplement with another name manually
        supplemental_names = {"insulin human": "insulin"}
        if node_name in supplemental_names:
            names[supplemental_names[node_name]] = 1

        # Return all the values after scrubbing
        scrubbed_values = {
            "node_curie": node_curie,
            "node_name": node_name,
            "node_category": node_category,
            "names": names,
        }
        return scrubbed_values

    # ############################################################################################
    def import_equivalencies(self):
        filename = "kg2_equivalencies.tsv"
        if not os.path.exists(filename):
            print(f"WARNING: Did not find equivalencies file {filename}. Skipping import")
            return
        print(f"INFO: Reading equivalencies from {filename}")
        filesize = os.path.getsize(filename)
        bytes_read = 0
        previous_percentage = -1
        problem_counter = 0

        kg_curies = self.kg_map["kg_curies"]
        kg_unique_concepts = self.kg_map["kg_unique_concepts"]

        stats = {
            "already equivalent": 0,
            "add new linked curie": 0,
            "neither curie found": 0,
            "association conflict": 0,
        }

        iline = 0
        with open(filename) as infile:
            for line in infile:
                bytes_read += len(line)

                #### Skip the column titles
                if iline == 0 and "n1.id" in line:
                    iline += 1
                    continue

                #### Strip and skip blank lines
                line = line.strip()
                match = re.match(r"\s*$", line)
                if match:
                    continue
                iline += 1

                # print(f"line={line}")
                columns = line.split("\t")
                node1_curie = columns[0]
                node2_curie = columns[1]
                # print(f"{node1_curie}  -  {node2_curie}")

                uc_node1_curie = node1_curie.upper()
                uc_node2_curie = node2_curie.upper()

                linking_curie = None
                uc_linking_curie = None
                done = False

                if uc_node1_curie in kg_curies:
                    linking_curie = node1_curie
                    uc_linking_curie = linking_curie.upper()
                    second_curie = node2_curie
                    uc_second_curie = second_curie.upper()

                elif uc_node2_curie in kg_curies:
                    linking_curie = node2_curie
                    uc_linking_curie = linking_curie.upper()
                    second_curie = node1_curie
                    uc_second_curie = second_curie.upper()

                else:
                    # print(f"ERROR: Niether {uc_node1_curie} nor {uc_node2_curie} found in kg_curies at line {iline+1}")
                    stats["neither curie found"] += 1
                    done = True

                if not done:
                    uc_linking_unique_concept_curie = kg_curies[uc_linking_curie]["uc_unique_concept_curie"]
                    linking_category = kg_curies[uc_linking_curie]["category"]

                    if uc_second_curie in kg_curies:
                        uc_second_unique_concept_curie = kg_curies[uc_second_curie]["uc_unique_concept_curie"]
                        if "KG2equivs" not in kg_curies[uc_second_curie]["source"]:
                            kg_curies[uc_second_curie]["source"] += ",KG2equivs"

                        #### If the two kg_curies already share the same unique concept
                        if uc_linking_unique_concept_curie == uc_second_unique_concept_curie:
                            stats["already equivalent"] += 1

                        #### But if they have different unique concepts, this potentially a problem, but one that will be cleaned up later probably
                        else:
                            stats["association conflict"] += 1
                            # print(f"WARNING: Association conflict: {linking_curie}->{uc_linking_unique_concept_curie} and {second_curie}->{uc_second_unique_concept_curie}")
                            kg_unique_concepts[uc_linking_unique_concept_curie]["all_uc_curies"][uc_second_curie] = (
                                "KG2equivs"
                            )
                            kg_unique_concepts[uc_linking_unique_concept_curie]["all_uc_curies"][
                                uc_second_unique_concept_curie
                            ] = "KG2equivs"

                            # This probably isn't needed, but things are not working the way that I want. FIXME
                            # kg_unique_concepts[uc_second_unique_concept_curie]['all_uc_curies'][uc_linking_curie] = 1
                            # kg_unique_concepts[uc_second_unique_concept_curie]['all_uc_curies'][uc_linking_unique_concept_curie] = 1

                    else:
                        print(f"Adding a new curie based on line '{line}'")
                        problem_counter += 1
                        # if problem_counter > 100:
                        #    exit()
                        stats["add new linked curie"] += 1
                        kg_curies[uc_second_curie] = {
                            "curie": second_curie,
                            "uc_unique_concept_curie": uc_linking_unique_concept_curie,
                            "name": None,
                            "full_name": None,
                            "category": linking_category,
                            "normalizer_name": None,
                            "normalizer_category": None,
                            "source": "KG2equivs",
                        }
                        kg_unique_concepts[uc_linking_unique_concept_curie]["all_uc_curies"][uc_second_curie] = (
                            "KG2equivs"
                        )

                percentage = int(bytes_read * 100.0 / filesize)
                if percentage > previous_percentage:
                    previous_percentage = percentage
                    print(str(percentage) + "%..", end="", flush=True)

        print("")
        print(f"INFO: Read {iline} equivalencies from {filename}")
        for stat_name, stat in stats.items():
            print(f"      - {stat_name}: {stat}")
        return

    # ############################################################################################
    def import_synonyms(self):
        filename = "kg2_synonyms.json"
        if not os.path.exists(filename):
            print(f"WARNING: Did not find synonyms file {filename}. Skipping import")
            return
        print(f"INFO: Reading synonyms from {filename}")

        kg_curies = self.kg_map["kg_curies"]
        kg_unique_concepts = self.kg_map["kg_unique_concepts"]
        kg_names = self.kg_map["kg_names"]
        kg_name_curies = self.kg_map["kg_name_curies"]

        stats = {"curie_found": 0, "curie_not_found": 0}

        with open(filename) as infile:
            node_synonyms = json.load(infile)
            inode = 0
            for curie, curie_names in node_synonyms.items():
                # print(f"{node} has synonyms {node_data}")
                inode += 1

                uc_curie = curie.upper()
                if uc_curie in kg_curies:
                    stats["curie_found"] += 1

                    uc_unique_concept_curie = kg_curies[uc_curie]["uc_unique_concept_curie"]

                    for name in curie_names:
                        # print(f"- {uc_curie} is also {name}")
                        lc_name = name.lower()
                        combined_key = lc_name + "---" + uc_curie

                        #### If we don't have this name yet, add it
                        if lc_name not in kg_names:
                            kg_names[lc_name] = {
                                "name": name,
                                "uc_unique_concept_curie": uc_unique_concept_curie,
                                "source": "KG2syn",
                                "uc_unique_concept_curies": {uc_unique_concept_curie: True},
                            }
                            kg_unique_concepts[uc_unique_concept_curie]["all_lc_names"][lc_name] = True

                        #### If this provenance record is not there yet, add it
                        if combined_key not in kg_name_curies:
                            kg_name_curies[combined_key] = {
                                "name": name,
                                "uc_curie": uc_curie,
                                "uc_unique_concept_curie": uc_unique_concept_curie,
                                "source": "KG2syn",
                            }

                else:
                    # print(f"ERROR: {filename} has a curie {curie} that was not previously recorded!")
                    stats["curie_not_found"] += 1

        print("")
        print(f"INFO: Read {inode} synonyms from {filename}")
        for stat_name, stat in stats.items():
            print(f"      - {stat_name}: {stat}")
        return

    #############################################################################################
    #### Go through all unique concepts and reset the lead unique_concept out of all the options based on a set of rules
    def reprioritize_unique_concepts(self):
        print("INFO: Reprioritizing the unique concepts leaders...")
        kg_nodes = self.kg_map["kg_nodes"]
        kg_unique_concepts = self.kg_map["kg_unique_concepts"]
        kg_curies = self.kg_map["kg_curies"]
        kg_names = self.kg_map["kg_names"]
        kg_name_curies = self.kg_map["kg_name_curies"]

        outfile = open("Problems.tsv", "w")

        concept_remap = {}

        debug_flag = DEBUG

        #### Progress tracking
        counter = 0
        n_items = len(kg_unique_concepts)
        previous_percentage = 0

        #### Loop over all unique_concepts to see if they need coalescing
        print("INFO: Looping over all unique_concepts...")
        for uc_unique_concept_curie, unique_concept in kg_unique_concepts.items():
            curie = unique_concept["curie"]
            curie_prefix = curie.split(":")[0].upper()

            #### Don't do anything fancy to meta nodes
            if curie_prefix == "BIOLINK":
                continue

            concept = {
                "category": unique_concept["category"],
                "name": unique_concept["name"],
                "all_categories": {},
                "all_curie_prefixes": {},
                "best_curie_score": -1,
                "best_curie": unique_concept["curie"],
                "best_category": unique_concept["category"],
                "best_name": unique_concept["name"],
            }
            manual_exception = False

            if debug_flag:
                print("===============================================")
                print(f"Considering {uc_unique_concept_curie}")

            for related_uc_curie in unique_concept["all_uc_curies"]:
                if related_uc_curie in kg_nodes:
                    node_category = kg_nodes[related_uc_curie]["category"]
                    node_curie = kg_nodes[related_uc_curie]["curie"]
                    node_name = kg_nodes[related_uc_curie]["adjusted_name"]
                    node_full_name = kg_nodes[related_uc_curie]["full_name"]
                    node_curie_prefix = node_curie.split(":")[0].upper()

                    ignore_category = False
                    if node_curie_prefix == "OMIM":
                        ignore_category = True

                    if not ignore_category:
                        if node_category not in concept["all_categories"]:
                            concept["all_categories"][node_category] = 0
                        concept["all_categories"][node_category] += 1

                    if node_curie_prefix not in concept["all_curie_prefixes"]:
                        concept["all_curie_prefixes"][node_curie_prefix] = 0
                    concept["all_curie_prefixes"][node_curie_prefix] += 1

                    this_score = 0
                    if node_curie_prefix in self.uc_curie_prefix_scores:
                        this_score = self.uc_curie_prefix_scores[node_curie_prefix]
                    if this_score > concept["best_curie_score"]:
                        concept["best_curie_score"] = this_score
                        concept["best_curie"] = node_curie
                        concept["best_category"] = node_category
                        concept["best_name"] = node_name

                    if node_curie_prefix == "NCBIGENE" and node_full_name.startswith("Genetic locus associated with"):
                        manual_exception = True
                        concept["best_curie_score"] = 9999
                        concept["best_curie"] = node_curie
                        concept["best_category"] = node_category
                        concept["best_name"] = node_full_name

                    if debug_flag:
                        print(f"  - After considering related {related_uc_curie}, concept = {concept}")

            drug_score = 0
            disease_score = 0
            protein_score = 0
            if (
                "biolink:Drug" in concept["all_categories"]
                or "biolink:ChemicalEntity" in concept["all_categories"]
                or "biolink:SmallMolecule" in concept["all_categories"]
                or "biolink:MolecularEntity" in concept["all_categories"]
            ):
                drug_score = 1
            if (
                "biolink:Disease" in concept["all_categories"]
                or "biolink:PhenotypicFeature" in concept["all_categories"]
                or "biolink:DiseaseOrPhenotypicFeature" in concept["all_categories"]
            ):
                disease_score = 1
            if (
                "biolink:Gene" in concept["all_categories"]
                or "biolink:Protein" in concept["all_categories"]
                or "biolink:GeneOrProtein" in concept["all_categories"]
                or "biolink:GenomicEntity" in concept["all_categories"]
            ):
                protein_score = 1

            #### Looks for concepts that are both a protein and a disease. A sign of trouble
            if protein_score > 0 and disease_score > 0 and not manual_exception:
                if True:
                    print("==== Protein-Disease CONFLICT! ===================================")
                    print(f"{uc_unique_concept_curie} '{concept['name']}' is a {concept['category']}")
                    print(f"  concept = {concept}")
                    outfile.write(
                        "\t".join(
                            [
                                uc_unique_concept_curie,
                                concept["name"],
                                concept["category"],
                            ]
                        )
                        + "\n"
                    )

            if drug_score > 0 and disease_score > 0 and not manual_exception:
                if (
                    "CHEMBL.COMPOUND" in concept["all_curie_prefixes"]
                    or "CHEBI" in concept["all_curie_prefixes"]
                    or "DRUGBANK" in concept["all_curie_prefixes"]
                    or "RXNORM" in concept["all_curie_prefixes"]
                    or "VANDF" in concept["all_curie_prefixes"]
                ):
                    drug_score += 1
                if "MONDO" in concept["all_curie_prefixes"] or "DOID" in concept["all_curie_prefixes"]:
                    disease_score += 1

                if drug_score > disease_score:
                    final_category = "biolink:Drug"
                elif disease_score > drug_score:
                    final_category = "biolink:Disease"
                elif disease_score == 1:
                    final_category = "ambiguous"
                else:
                    final_category = "CONFLICT"

                is_problem = False
                if (
                    final_category == "biolink:Drug"
                    and concept["best_category"] != "biolink:Drug"
                    and concept["best_category"] != "biolink:ChemicalEntity"
                    and concept["best_category"] != "biolink:MolecularEntity"
                    and concept["best_category"] != "biolink:SmallMolecule"
                ):
                    is_problem = True
                if (
                    final_category == "biolink:Disease"
                    and concept["best_category"] != "biolink:PhenotypicFeature"
                    and concept["best_category"] != "biolink:DiseaseOrPhenotypicFeature"
                ):
                    is_problem = True
                if final_category == "CONFLICT":
                    is_problem = True

                if is_problem:
                    print("***** PROBLEM ***************************")
                    print(f"{uc_unique_concept_curie} '{concept['name']}' is a {concept['category']}")
                    print(f"  concept = {concept}")
                    print(f"  drug_score={drug_score}, disease_score={disease_score}, final_category={final_category}")
                    outfile.write(
                        "\t".join(
                            [
                                uc_unique_concept_curie,
                                concept["name"],
                                concept["category"],
                            ]
                        )
                        + "\n"
                    )

            if debug_flag:
                print("========================================================")
                print(f"{uc_unique_concept_curie} '{concept['name']}' is a {concept['category']}")
                print(f"  concept = {concept}")

            #### Record the necessary remapping
            if concept["best_curie"].upper() != uc_unique_concept_curie:
                concept_remap[uc_unique_concept_curie] = concept["best_curie"].upper()

            #### Update the kg_unique_concept with the final normalized information
            unique_concept["curie"] = concept["best_curie"]
            unique_concept["name"] = concept["best_name"]
            unique_concept["category"] = concept["best_category"]

            #### Show progress information
            counter += 1
            percentage = int(counter * 100.0 / n_items)
            if percentage > previous_percentage:
                previous_percentage = percentage
                print(str(percentage) + "%..", end="", flush=True)

        #### Remap kg_nodes
        for uc_curie, element in kg_nodes.items():
            if element["uc_unique_concept_curie"] in concept_remap:
                element["uc_unique_concept_curie"] = concept_remap[element["uc_unique_concept_curie"]]

        #### Remap kg_curies
        for uc_curie, element in kg_curies.items():
            if element["uc_unique_concept_curie"] in concept_remap:
                element["uc_unique_concept_curie"] = concept_remap[element["uc_unique_concept_curie"]]

        #### Remap kg_names
        for lc_name, element in kg_names.items():
            if element["uc_unique_concept_curie"] in concept_remap:
                element["uc_unique_concept_curie"] = concept_remap[element["uc_unique_concept_curie"]]

            #### Add in the new one                                                                         Maybe should be done, but not used any more anyway?
            # element['uc_unique_concept_curies'][element['uc_unique_concept_curie']] = True

            #### Remove any remapped uc_curies
            # new_uc_unique_concept_curies = {}
            # for uc_curie in element['uc_unique_concept_curies']:
            #    if uc_curie not in new_concept_remap:
            #        new_uc_unique_concept_curies[uc_curie] = True
            # element['uc_unique_concept_curies'] = new_uc_unique_concept_curies

        #### Remap kg_name_curies
        for lc_name_uc_curie, element in kg_name_curies.items():
            if element["uc_unique_concept_curie"] in concept_remap:
                element["uc_unique_concept_curie"] = concept_remap[element["uc_unique_concept_curie"]]

        #### And update the unique_concepts
        for uc_curie, uc_new_curie in concept_remap.items():
            kg_unique_concepts[uc_new_curie] = kg_unique_concepts[uc_curie]
            del kg_unique_concepts[uc_curie]

        outfile.close()

    # ############################################################################################
    # This is just for testing. Doesn't actually do anything
    def update_categories(self):
        print("INFO: Scanning through unique_concepts, looking for problem nodes to fix")

        debug = DEBUG

        # Get all the unique_concept_curies
        cursor = self.connection.cursor()
        cursor.execute("SELECT uc_curie, category, name FROM unique_concepts LIMIT 1000000")
        unique_concept_rows = cursor.fetchall()

        batch_unique_concepts = {}
        counter = 0

        for unique_concept_row in unique_concept_rows:
            curie = unique_concept_row[0]
            curie_prefix = curie.split(":")[0].upper()
            if curie_prefix == "BIOLINK":
                continue

            batch_unique_concepts[unique_concept_row[0]] = {
                "category": unique_concept_row[1],
                "name": unique_concept_row[2],
                "all_categories": {},
                "all_curie_prefixes": {},
                "best_curie_score": 0,
                "best_curie": unique_concept_row[0],
                "best_category": unique_concept_row[1],
                "best_name": unique_concept_row[2],
            }

            if len(batch_unique_concepts) == 100 or unique_concept_row[0] == unique_concept_rows[-1][0]:
                # Create the SQL "IN" list
                quote_fixed_uc_curies_list = []
                for uc_curie in batch_unique_concepts:
                    uc_curie = re.sub(r"'", "''", uc_curie)  # Replace embedded ' characters with ''
                    quote_fixed_uc_curies_list.append(uc_curie)
                curies_list_str = "','".join(quote_fixed_uc_curies_list)

                # Get all the curies for these concepts and their categories
                sql = f"""
                    SELECT curie, unique_concept_curie, category, adjusted_name
                        FROM nodes
                        WHERE unique_concept_curie IN ( '{curies_list_str}' )"""
                cursor = self.connection.cursor()
                cursor.execute(sql)
                rows = cursor.fetchall()

                if debug:
                    print(".", end="", flush=True)

                for row in rows:
                    curie = row[0]
                    uc_unique_concept_curie = row[1]
                    node_category = row[2]
                    name = row[3]
                    curie_prefix = curie.split(":")[0].upper()

                    if node_category not in batch_unique_concepts[uc_unique_concept_curie]["all_categories"]:
                        batch_unique_concepts[uc_unique_concept_curie]["all_categories"][node_category] = 0
                    batch_unique_concepts[uc_unique_concept_curie]["all_categories"][node_category] += 1

                    if curie_prefix not in batch_unique_concepts[uc_unique_concept_curie]["all_curie_prefixes"]:
                        batch_unique_concepts[uc_unique_concept_curie]["all_curie_prefixes"][curie_prefix] = 0
                    batch_unique_concepts[uc_unique_concept_curie]["all_curie_prefixes"][curie_prefix] += 1

                    if (
                        curie_prefix in self.uc_curie_prefix_scores
                        and self.uc_curie_prefix_scores[curie_prefix]
                        > batch_unique_concepts[uc_unique_concept_curie]["best_curie_score"]
                    ):
                        batch_unique_concepts[uc_unique_concept_curie]["best_curie_score"] = (
                            self.uc_curie_prefix_scores[curie_prefix]
                        )
                        batch_unique_concepts[uc_unique_concept_curie]["best_curie"] = curie
                        batch_unique_concepts[uc_unique_concept_curie]["best_category"] = node_category
                        batch_unique_concepts[uc_unique_concept_curie]["best_name"] = name

                for uc_unique_concept_curie, concept in batch_unique_concepts.items():
                    drug_score = 0
                    disease_score = 0
                    protein_score = 0
                    if (
                        "biolink:Drug" in concept["all_categories"]
                        or "biolink:ChemicalEntity" in concept["all_categories"]
                        or "biolink:SmallMolecule" in concept["all_categories"]
                        or "biolink:MolecularEntity" in concept["all_categories"]
                    ):
                        drug_score = 1
                    if (
                        "biolink:Disease" in concept["all_categories"]
                        or "biolink:PhenotypicFeature" in concept["all_categories"]
                        or "biolink:DiseaseOrPhenotypicFeature" in concept["all_categories"]
                    ):
                        disease_score = 1
                    if (
                        "biolink:Gene" in concept["all_categories"]
                        or "biolink:Protein" in concept["all_categories"]
                        or "biolink:GeneOrProtein" in concept["all_categories"]
                        or "biolink:GenomicEntity" in concept["all_categories"]
                    ):
                        protein_score = 1

                    if (
                        False
                        and uc_unique_concept_curie != batch_unique_concepts[uc_unique_concept_curie]["best_curie"]
                    ):
                        print("***************************************************")
                        print(f"{uc_unique_concept_curie} is a {concept['category']}")
                        print(f"  concept = {concept}")
                        counter += 1
                        if counter >= 50:
                            exit()

                    if protein_score > 0 and disease_score > 0:
                        if True:
                            print("==== Protein-Disease CONFLICT! ===================================")
                            print(f"{uc_unique_concept_curie} '{concept['name']}' is a {concept['category']}")
                            print(f"  concept = {concept}")

                    if drug_score > 0 and disease_score > 0:
                        if debug:
                            print("========================================================")
                            print(f"{uc_unique_concept_curie} '{concept['name']}' is a {concept['category']}")
                            print(f"  concept = {concept}")

                        if (
                            "CHEMBL.COMPOUND" in concept["all_curie_prefixes"]
                            or "CHEBI" in concept["all_curie_prefixes"]
                            or "DRUGBANK" in concept["all_curie_prefixes"]
                            or "RXNORM" in concept["all_curie_prefixes"]
                        ):
                            drug_score += 1
                        if "MONDO" in concept["all_curie_prefixes"] or "DOID" in concept["all_curie_prefixes"]:
                            disease_score += 1

                        if drug_score > disease_score:
                            final_category = "biolink:Drug"
                        elif disease_score > drug_score:
                            final_category = "biolink:Disease"
                        elif disease_score == 1:
                            final_category = "ambiguous"
                        else:
                            final_category = "CONFLICT"

                        if debug:
                            print(
                                f"  drug_score={drug_score}, disease_score={disease_score}, final_category={final_category}"
                            )

                        counter += 1
                        if counter >= 50:
                            exit()

                #### Reset the batch
                batch_unique_concepts = {}

        print(f"INFO: Processed {len(unique_concept_rows)} unique concepts")

    # ############################################################################################

    # Access methods

    # ############################################################################################
    def get_curies_and_types(self, name, kg_name="KG2"):
        # Determine the table prefix for the knowledge graph selected
        if kg_name.upper() == "KG1" or kg_name.upper() == "KG2":
            kg_prefix = kg_name.lower()
        else:
            print("ERROR: kg_name must be either 'KG1' or 'KG2'")
            return None

        kg_prefix = "kg2"

        # Set up the return list
        curies_and_types = []

        # Search the synonym table for the provided name
        cursor = self.connection.cursor()
        cursor.execute(
            f"SELECT unique_concept_curie FROM {kg_prefix}_synonym{TESTSUFFIX} WHERE lc_name = ?",
            (name.lower(),),
        )
        rows = cursor.fetchall()

        # If no rows came back, see if it matches a CURIE
        if len(rows) == 0:
            cursor = self.connection.cursor()
            cursor.execute(
                f"SELECT unique_concept_curie FROM {kg_prefix}_curie{TESTSUFFIX} WHERE uc_curie = ?",
                (name.upper(),),
            )  # FIXME: need to make curie upper()?
            rows = cursor.fetchall()

        # If there are still no rows, then just return an empty list
        if len(rows) == 0:
            return curies_and_types

        # If multiple rows come back, this is probably an error in the database
        if len(rows) > 1:
            print(
                f"WARNING: Search in NodeSynonymizer for '{name}' turned up more than one unique_concept. This shouldn't be."
            )

        # Extract the CURIE for the unique concept
        unique_concept_curie = rows[0][0]

        # Get the list of nodes that link to this concept
        cursor = self.connection.cursor()
        cursor.execute(
            f"SELECT * FROM {kg_prefix}_node{TESTSUFFIX} WHERE unique_concept_curie = ?",
            (unique_concept_curie,),
        )
        rows = cursor.fetchall()
        curies_and_types = []
        for row in rows:
            curies_and_types.append({"curie": row[1], "type": row[4], "name": row[3]})
        return curies_and_types

    # ############################################################################################
    def get_curies_and_types_and_names(self, name, kg_name="KG2"):
        curies_and_types = self.get_curies_and_types(name, kg_name=kg_name)

        for entity in curies_and_types:
            # Try to fetch the description from the knowledge graph
            try:
                properties = RU.get_node_properties(entity["curie"])
                if "description" in properties:
                    entity["description"] = properties["description"]
            except:
                # This will happen with this node is in KG2 but not KG1. FIXME
                pass

        return curies_and_types

    # ############################################################################################
    def get_names(self, curie, kg_name="KG2"):
        # Determine the table prefix for the knowledge graph selected
        if kg_name.upper() == "KG1" or kg_name.upper() == "KG2":
            kg_prefix = kg_name.lower()
        else:
            print("ERROR: kg_name must be either 'KG1' or 'KG2'")
            return None

        # Set up the return list
        names = []

        # Search the curie table for the provided curie
        cursor = self.connection.cursor()
        cursor.execute(
            f"SELECT unique_concept_curie FROM {kg_prefix}_curie{TESTSUFFIX} WHERE uc_curie = ?",
            (curie.upper(),),
        )  # FIXME: need to make curie upper()?
        rows = cursor.fetchall()

        # If there are still no rows, then just return an empty list
        if len(rows) == 0:
            return names

        # If multiple rows come back, this is probably an error in the database
        if len(rows) > 1:
            print(
                f"WARNING: Search in NodeSynonymizer for '{curie}' turned up more than one unique_concept. This shouldn't be."
            )

        # Extract the CURIE for the unique concept
        unique_concept_curie = rows[0][0]

        # Get the list of names that link to this concept
        cursor = self.connection.cursor()
        cursor.execute(
            f"SELECT * FROM {kg_prefix}_synonym{TESTSUFFIX} WHERE unique_concept_curie = ?",
            (unique_concept_curie,),
        )
        rows = cursor.fetchall()
        for row in rows:
            names.append(row[1])
        return names

    # ############################################################################################
    def get_curies(self, name, kg_name="KG2"):
        curies_and_types = self.get_curies_and_types(name, kg_name)

        if curies_and_types is None:
            return None

        # Return a list of curies
        curies = []
        for curies_and_type in curies_and_types:
            curies.append(curies_and_type["curie"])
        return curies

    # ############################################################################################
    def is_curie_present(self, curie, kg_name="KG2"):
        # Determine the table prefix for the knowledge graph selected
        if kg_name.upper() == "KG1" or kg_name.upper() == "KG2":
            kg_prefix = kg_name.lower()
        else:
            print("ERROR: kg_name must be either 'KG1' or 'KG2'")
            return None

        cursor = self.connection.cursor()
        cursor.execute(
            f"SELECT * FROM {kg_prefix}_node{TESTSUFFIX} WHERE uc_curie = ?",
            (curie.upper(),),
        )
        rows = cursor.fetchall()

        if len(rows) == 0:
            return False
        return True

    # ############################################################################################
    def get_KG1_curies(self, name):
        curies_list = self.get_curies(name, kg_name="KG2")
        return curies_list

    # ############################################################################################
    def convert_curie(self, curie, namespace):
        kg_name = "KG2"  # FIXME. hardwired to KG1 for now
        curies_and_types = self.get_curies_and_types(curie, kg_name)

        if len(curies_and_types) == 0:
            return []

        curies = {}
        curies_list = []
        for row in curies_and_types:
            curie = row["curie"]
            match = re.match(namespace + ":", curie)
            if match:
                if curie not in curies:
                    curies_list.append(curie)
                    curies[curie] = 1
        return curies_list

    # ############################################################################################
    def get_equivalent_nodes(self, curies, return_all_categories=False):
        return self.get_canonical_curies(
            curies,
            return_all_categories=return_all_categories,
            return_type="equivalent_nodes",
        )

    # ############################################################################################
    def get_canonical_curies(
        self,
        curies=None,
        names=None,
        return_all_categories=False,
        return_type="canonical_curies",
    ):
        # If the provided curies or names is just a string, turn it into a list
        if isinstance(curies, str):
            curies = [curies]
        if isinstance(names, str):
            names = [names]

        # Set up containers for the batches and results
        batches = []
        results = {}

        # Set up the category manager
        category_manager = CategoryManager()

        # Make sets of comma-separated list strings for the curies and set up the results dict with all the input values
        uc_curies = []
        curie_map = {}
        batch_size = 0
        if curies is not None:
            for curie in curies:
                if curie is None:
                    continue
                results[curie] = None
                uc_curie = curie.upper()
                curie_map[uc_curie] = curie
                uc_curie = re.sub(r"'", "''", uc_curie)  # Replace embedded ' characters with ''
                uc_curies.append(uc_curie)
                batch_size += 1
                if batch_size > 5000:
                    batches.append({"batch_type": "curies", "batch_str": "','".join(uc_curies)})
                    uc_curies = []
                    batch_size = 0
            if batch_size > 0:
                batches.append({"batch_type": "curies", "batch_str": "','".join(uc_curies)})

        # Make sets of comma-separated list strings for the names
        lc_names = []
        name_map = {}
        batch_size = 0
        if names is not None:
            for name in names:
                if name is None:
                    continue
                results[name] = None
                lc_name = name.lower()
                name_map[lc_name] = name
                lc_name = re.sub(r"'", "''", lc_name)  # Replace embedded ' characters with ''
                lc_names.append(lc_name)
                batch_size += 1
                if batch_size > 5000:
                    batches.append({"batch_type": "names", "batch_str": "','".join(lc_names)})
                    lc_names = []
                    batch_size = 0
            if batch_size > 0:
                batches.append({"batch_type": "names", "batch_str": "','".join(lc_names)})

        for batch in batches:
            # print(f"INFO: Batch {i_batch} of {batch['batch_type']}")
            # i_batch += 1
            if batch["batch_type"] == "curies":
                if return_type == "equivalent_nodes":
                    sql = f"""
                        SELECT C.curie,C.unique_concept_curie,N.curie,N.category,U.category
                          FROM curies AS C
                         INNER JOIN nodes AS N ON C.unique_concept_curie == N.unique_concept_curie
                         INNER JOIN unique_concepts AS U ON C.unique_concept_curie == U.uc_curie
                         WHERE C.uc_curie in ( '{batch["batch_str"]}' )"""
                else:
                    sql = f"""
                        SELECT C.curie,C.unique_concept_curie,U.curie,U.name,U.category
                          FROM curies AS C
                         INNER JOIN unique_concepts AS U ON C.unique_concept_curie == U.uc_curie
                         WHERE C.uc_curie in ( '{batch["batch_str"]}' )"""
            else:
                sql = f"""
                    SELECT S.name,S.unique_concept_curie,U.curie,U.name,U.category
                      FROM names AS S
                     INNER JOIN unique_concepts AS U ON S.unique_concept_curie == U.uc_curie
                     WHERE S.lc_name in ( '{batch["batch_str"]}' )"""
            # print(f"INFO: Processing {batch['batch_type']} batch: {batch['batch_str']}")
            cursor = self.connection.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()

            # Loop through all rows, building the list
            batch_curie_map = {}
            for row in rows:
                # If the curie or name is not found in results, try to use the curie_map{}/name_map{} to resolve capitalization issues
                entity = row[0]
                if entity not in results:
                    if batch["batch_type"] == "curies":
                        if entity.upper() in curie_map:
                            entity = curie_map[entity.upper()]
                    else:
                        if entity.lower() in name_map:
                            entity = name_map[entity.lower()]

                # Now store this curie in the list
                if entity in results:
                    if row[1] not in batch_curie_map:
                        batch_curie_map[row[1]] = {}
                    batch_curie_map[row[1]][entity] = 1

                    # If the return turn is equivalent_nodes, then add the node curie to the dict
                    if return_type == "equivalent_nodes":
                        if results[entity] is None:
                            results[entity] = {}
                        node_curie = row[2]
                        results[entity][node_curie] = row[3]

                    # Else the return type is assumed to be the canonical node
                    else:
                        results[entity] = {
                            "preferred_curie": row[2],
                            "preferred_name": row[3],
                            "preferred_category": row[4],
                        }

                    #### Also store tidy categories
                    if return_all_categories:
                        results[entity]["expanded_categories"] = category_manager.get_expansive_categories(row[4])

                else:
                    print(f"ERROR: Unable to find entity {entity}")

            # If all_categories were requested, do another query for those
            if return_all_categories:
                # Create the SQL IN list
                uc_curies_list = []
                for uc_curie in batch_curie_map:
                    uc_curie = re.sub(r"'", "''", uc_curie)  # Replace embedded ' characters with ''
                    uc_curies_list.append(uc_curie)
                curies_list_str = "','".join(uc_curies_list)

                # Get all the curies for these concepts and their categories
                sql = f"""
                    SELECT curie,unique_concept_curie,category
                      FROM curies
                     WHERE unique_concept_curie IN ( '{curies_list_str}' )"""
                cursor = self.connection.cursor()
                cursor.execute(sql)
                rows = cursor.fetchall()

                entity_all_categories = {}
                for row in rows:
                    uc_unique_concept_curie = row[1]
                    node_category = row[2]
                    entities = batch_curie_map[uc_unique_concept_curie]

                    #### Eric says: I'm a little concerned that this entity is stomping on the previous entity. What's really going on here? FIXME
                    for entity in entities:
                        # Now store this category in the list
                        if entity in results:
                            if entity not in entity_all_categories:
                                entity_all_categories[entity] = {}
                            if node_category is None:
                                continue
                            if node_category not in entity_all_categories[entity]:
                                entity_all_categories[entity][node_category] = 0
                            entity_all_categories[entity][node_category] += 1
                        else:
                            print(f"ERROR: Unable to find entity {entity}")

                # Now store the final list of categories into the list
                for entity, all_categories in entity_all_categories.items():
                    if entity in results and results[entity] is not None:
                        results[entity]["all_categories"] = all_categories

        return results

    # ############################################################################################
    # Return results in the Node Normalizer format, either from SRI or KG1 or KG2
    def get_normalizer_results(self, entities=None):
        # If no entity was passed, then nothing to do
        if entities is None:
            return None

        # If the provided value is just a string, turn it into a list
        if isinstance(entities, str):
            entities = [entities]

        # Loop over all entities and get the results
        results = {}
        for entity in entities:
            # Search the curie table for the provided entity
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT unique_concept_curie FROM curies WHERE uc_curie = ?",
                (entity.upper(),),
            )
            rows = cursor.fetchall()

            # If no rows came back, see if it matches a name
            if len(rows) == 0:
                cursor = self.connection.cursor()
                cursor.execute(
                    "SELECT unique_concept_curie FROM names WHERE lc_name = ?",
                    (entity.lower(),),
                )
                rows = cursor.fetchall()

            # If there are still no rows, then just move on
            if len(rows) == 0:
                results[entity] = None
                continue

            # If multiple rows come back, this is probably an error in the database
            if len(rows) > 1:
                print(
                    f"ERROR: Search in NodeSynonymizer for '{entity}' turned up more than one unique_concept. This shouldn't be."
                )

            # Extract the CURIE for the unique concept
            unique_concept_curie = rows[0][0]

            # Get the list of nodes that link to this concept
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT * FROM nodes WHERE unique_concept_curie = ?",
                (unique_concept_curie,),
            )
            rows = cursor.fetchall()
            nodes = []
            for row in rows:
                nodes.append(
                    {
                        "identifier": row[1],
                        "category": row[5],
                        "label": row[3],
                        "original_label": row[2],
                    }
                )

            # Get the list of curies that link to this concept
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT * FROM curies WHERE unique_concept_curie = ?",
                (unique_concept_curie,),
            )
            rows = cursor.fetchall()
            curies = []
            categories = {}
            names = {}
            for row in rows:
                #### Store the curies
                curies.append(
                    {
                        "identifier": row[1],
                        "name": row[3],
                        "full_name": row[4],
                        "category": row[5],
                        "normalizer_name": row[6],
                        "normalizer_category": row[7],
                        "source": row[8],
                    }
                )

                #### Store the categories
                category = row[5]
                if category == "":
                    category = None
                if category is not None:
                    if category not in categories:
                        categories[category] = 0
                    categories[category] += 1
                normalizer_category = row[7]
                if normalizer_category == "":
                    normalizer_category = None
                if normalizer_category is not None:
                    if category is None or normalizer_category != category:
                        if normalizer_category not in categories:
                            categories[normalizer_category] = 0
                        categories[normalizer_category] += 1

                #### Store the names
                name = row[3]
                if name == "":
                    name = None
                if name is not None:
                    if name not in names:
                        names[name] = 0
                    names[name] += 1
                normalizer_name = row[6]
                if normalizer_name == "":
                    normalizer_name = None
                if normalizer_name is not None:
                    if name is None or normalizer_name != name:
                        if normalizer_name not in names:
                            names[normalizer_name] = 0
                        names[normalizer_name] += 1
                full_name = row[4]
                if full_name == "":
                    full_name = None
                if full_name is not None and full_name != "":
                    if full_name != name:
                        if normalizer_name is None or normalizer_name != full_name:
                            if full_name not in names:
                                names[full_name] = 0
                            names[full_name] += 1

            # Get the list of synonyms that link to this concept
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT * FROM names WHERE unique_concept_curie = ?",
                (unique_concept_curie,),
            )
            rows = cursor.fetchall()
            synonyms = []
            for row in rows:
                synonyms.append({"name": row[1], "source": row[3]})

            # Get the unique concept information
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT * FROM unique_concepts WHERE uc_curie = ?",
                (unique_concept_curie,),
            )
            rows = cursor.fetchall()

            # If multiple rows come back, this is probably an error in the database
            if len(rows) > 1:
                print(
                    f"ERROR: Search in NodeSynonymizer for '{unique_concept_curie}' turned up more than one unique_concept. This shouldn't be."
                )

            # Fill in the unique identifier
            row = rows[0]
            id = {
                "identifier": row[1],
                "name": row[2],
                "category": row[3],
                "SRI_normalizer_curie": row[4],
                "SRI_normalizer_name": row[5],
                "SRI_normalizer_category": row[6],
            }

            # Get the synonym provenance information
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT * FROM name_curies WHERE unique_concept_curie = ?",
                (unique_concept_curie,),
            )
            rows = cursor.fetchall()
            synonym_provenance = []
            for row in rows:
                synonym_provenance.append({"name": row[1], "uc_curie": row[2], "source": row[4]})

            # Add this entry to the final results dict
            results[entity] = {
                "nodes": nodes,
                "equivalent_identifiers": curies,
                "synonyms": names,
                "synonym_provenance": synonym_provenance,
                "id": id,
                "categories": categories,
            }

        return results

    # ############################################################################################
    def get_total_entity_count(self, node_type, kg_name="KG1"):
        # Just get a count of all unique_concepts
        cursor = self.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM unique_concepts WHERE category = ?", (node_type,))
        rows = cursor.fetchall()

        # Return the count value
        return rows[0][0]

    # ############################################################################################
    def test_query(self):
        cursor = self.connection.cursor()
        # cursor.execute( f"SELECT TOP 10 * FROM {kg_prefix}_synonym{TESTSUFFIX} WHERE synonym = ?", (name.upper(),) )
        # cursor.execute( f"SELECT * FROM {kg_prefix}_synonym{TESTSUFFIX} LIMIT 100 ")
        # cursor.execute( f"SELECT * FROM {kg_prefix}_curie{TESTSUFFIX} LIMIT 100 ")
        # cursor.execute( f"SELECT * FROM {kg_prefix}_node{TESTSUFFIX} LIMIT 100 ")
        # cursor.execute( f"SELECT * FROM unique_concepts WHERE kg2_best_curie IS NULL LIMIT 100 ")
        # cursor.execute( f"""
        #    SELECT C.curie,C.unique_concept_curie,N.curie,N.kg_presence FROM {kg_prefix}_curie{TESTSUFFIX} AS C
        #     INNER JOIN {kg_prefix}_node{TESTSUFFIX} AS N ON C.unique_concept_curie == N.unique_concept_curie
        #     WHERE C.uc_curie in ( 'DOID:384','DOID:13636' )""" )
        cursor.execute("SELECT * FROM unique_concepts LIMIT 100")

        rows = cursor.fetchall()
        for row in rows:
            print(row)


# ############################################################################################
def run_example_1():
    synonymizer = NodeSynonymizer()

    print("==== Testing for finding curies by name ====")
    tests = [
        "APS2",
        "phenylketonuria",
        "Gaucher's disease",
        "Gauchers disease",
        "Gaucher disease",
        "Alzheimer Disease",
        "Alzheimers disease",
        "Alzheimer's Disease",
        "kidney",
        "KIDney",
        "P06865",
        "HEXA",
        "UniProtKB:P12004",
        "rickets",
        "fanconi anemia",
        "retina",
        "is",
        "insulin",
    ]

    # The first one takes a bit longer, so do one before starting the timer
    test = synonymizer.get_curies("ibuprofen")

    t0 = timeit.default_timer()
    for test in tests:
        curies = synonymizer.get_curies(test)
        print(test + " = " + str(curies))
    t1 = timeit.default_timer()
    print("Elapsed time: " + str(t1 - t0))


# ############################################################################################
def run_example_2():
    synonymizer = NodeSynonymizer()

    print("==== Testing presence of CURIEs ============================")
    tests = [
        "REACT:R-HSA-2160456",
        "DOID:9281",
        "OMIM:261600",
        "DOID:1926xx",
        "HP:0002511",
        "UBERON:0002113",
        "UniProtKB:P06865",
        "P06865",
        "KEGG:C10399",
        "GO:0034187",
        "DOID:10652xx",
    ]

    t0 = timeit.default_timer()
    for test in tests:
        is_present = synonymizer.is_curie_present(test)
        print(test + " = " + str(is_present))
    t1 = timeit.default_timer()
    print("Elapsed time: " + str(t1 - t0))


# ############################################################################################
def run_example_3():
    synonymizer = NodeSynonymizer()

    print("==== Getting properties by CURIE ============================")
    tests = [
        "REACT:R-HSA-2160456",
        "DOID:9281",
        "OMIM:261600",
        "DOID:1926xx",
        "P06865",
        "UNIProtKB:P01308",
    ]

    t0 = timeit.default_timer()
    for test in tests:
        node_properties = synonymizer.get_curies_and_types_and_names(test)
        print(test + " = " + str(node_properties))
    t1 = timeit.default_timer()
    print("Elapsed time: " + str(t1 - t0))


# ############################################################################################
def run_example_4():
    synonymizer = NodeSynonymizer()

    print("==== Testing for KG1 and KG2 ============================")
    tests = [
        "APS2",
        "phenylketonuria",
        "Gauchers disease",
        "kidney",
        "HEXA",
        "UniProtKB:P12004",
        "fanconi anemia",
        "ibuprofen",
    ]

    t0 = timeit.default_timer()
    for test in tests:
        curies = synonymizer.get_curies(test)
        print(test + " in KG1 = " + str(curies))
        # curies = synonymizer.get_curies(test, kg_name='KG2')
        # print(test+" in KG2 = "+str(curies))
    t1 = timeit.default_timer()
    print("Elapsed time: " + str(t1 - t0))


# ############################################################################################
def run_example_5():
    synonymizer = NodeSynonymizer()

    print("==== Getting KG1 CURIEs ============================")
    tests = [
        "UMLS:C0031485",
        "UMLS:C0017205",
        "UniProtKB:P06865",
        "MESH:D005199",
        "HEXA",
        "CHEBI:5855",
        "fanconi anemia",
        "ibuprofen",
        "DOID:9281",
    ]

    t0 = timeit.default_timer()
    for test in tests:
        curies = synonymizer.get_KG1_curies(test)
        print(test + " = " + str(curies))
    t1 = timeit.default_timer()
    print("Elapsed time: " + str(t1 - t0))


# ############################################################################################
def run_example_6():
    synonymizer = NodeSynonymizer()

    print("==== Convert CURIEs to requested namespace ============================")
    tests = [
        ["UMLS:C0031485", "DOID"],
        ["FMA:7203", "UBERON"],
        ["MESH:D005199", "DOID"],
        ["CHEBI:5855", "CHEMBL.COMPOUND"],
        ["ibuprofen", "UMLS"],
    ]

    t0 = timeit.default_timer()
    for test in tests:
        curies = synonymizer.convert_curie(test[0], test[1])
        print(f"{test[0]} -> {test[1]} = " + str(curies))
    t1 = timeit.default_timer()
    print("Elapsed time: " + str(t1 - t0))


# ############################################################################################
def run_example_6b():
    synonymizer = NodeSynonymizer()

    print("==== Get all equivalent nodes in a KG for an input curie ============================")
    tests = [
        "DOID:14330",
        "UMLS:C0031485",
        "FMA:7203",
        "MESH:D005199",
        "CHEBI:5855",
        "DOID:9281",
    ]
    # tests = [ "DOID:9281" ]

    t0 = timeit.default_timer()
    for test in tests:
        print("--- KG1 ---")
        nodes = synonymizer.get_equivalent_nodes(test, kg_name="KG1")
        print(f"{test} = " + str(nodes))
        print("--- KG2 ---")
        nodes = synonymizer.get_equivalent_nodes(test, kg_name="KG2")
        print(f"{test} = " + str(nodes))
        print()
        # print(json.dumps(nodes,sort_keys=True,indent=2))
    t1 = timeit.default_timer()
    print("Elapsed time: " + str(t1 - t0))


# ############################################################################################
def run_example_7():
    synonymizer = NodeSynonymizer()

    for kg_name in ["KG1", "KG2"]:
        print(f"==== Get total number of concepts for several types for {kg_name} ============================")
        t0 = timeit.default_timer()
        for entity_type in [
            "biolink:ChemicalEntity",
            "biolink:Drug",
            "biolink:Disease",
            "biolink:Protein",
            "biolink:Gene",
            "cheesecake",
        ]:
            print(f"count({entity_type}) = {synonymizer.get_total_entity_count(entity_type, kg_name=kg_name)}")
        t1 = timeit.default_timer()
        print("Elapsed time: " + str(t1 - t0))


# ############################################################################################
def run_example_8():
    synonymizer = NodeSynonymizer()

    print("==== Test SELECT ============================")
    synonymizer.test_select("phenylketonuria")
    # synonymizer.test_select('UMLS:C4710278')
    # synonymizer.test_select('UniProtKB:P06865')
    # print(synonymizer.is_curie_present('UMLS:C4710278'))


# ############################################################################################
def run_example_9():
    synonymizer = NodeSynonymizer()
    import copy

    print("==== Get canonical curies for a set of input curies ============================")
    curies = [
        "DOID:14330",
        "UMLS:C0031485",
        "FMA:7203",
        "MESH:D005199",
        "CHEBI:5855",
        "DOID:9281xxxxx",
        "MONDO:0005520",
    ]
    names = [
        "phenylketonuria",
        "ibuprofen",
        "P06865",
        "HEXA",
        "Parkinson's disease",
        "supernovas",
        "Bob's Uncle",
        'double "quotes"',
        None,
    ]
    # curies = [ "UMLS:C0002371", "UMLS:C0889200" ]

    combined_list = copy.copy(curies)
    combined_list.extend(names)

    t0 = timeit.default_timer()
    canonical_curies = synonymizer.get_canonical_curies(curies=curies, return_all_categories=True)
    t1 = timeit.default_timer()
    canonical_curies2 = synonymizer.get_canonical_curies(names=names, return_all_categories=True)
    t2 = timeit.default_timer()
    canonical_curies3 = synonymizer.get_canonical_curies(
        curies=combined_list,
        names=combined_list,
        return_all_categories=True,
        return_type="equivalent_nodes",
    )
    t3 = timeit.default_timer()
    # print(json.dumps(canonical_curies,sort_keys=True,indent=2))
    # print("Elapsed time: "+str(t1-t0))
    # print(json.dumps(canonical_curies2,sort_keys=True,indent=2))
    # print("Elapsed time: "+str(t2-t1))
    print(json.dumps(canonical_curies3, sort_keys=True, indent=2))
    print("Elapsed time: " + str(t3 - t2))


# ############################################################################################
def run_example_10():
    synonymizer = NodeSynonymizer()

    print("==== Complex name query ============================")
    node_ids = [
        "CHEMBL.MECHANISM:potassium_channel,_inwardly_rectifying,_subfamily_j,_member_11_opener",
        "CHEMBL.MECHANISM:potassium_channel,_inwardly_rectifying,_subfamily_j,_member_8_opener",
        "CHEMBL.MECHANISM:endothelin_receptor,_et-a/et-b_antagonist",
        "CHEMBL.MECHANISM:amylin_receptor_amy1,_calcr/ramp1_agonist",
        "CHEMBL.MECHANISM:sulfonylurea_receptor_2,_kir6.2_opener",
        "CHEMBL.MECHANISM:sulfonylurea_receptor_1,_kir6.2_blocker",
        "CHEMBL.MECHANISM:amiloride-sensitive_sodium_channel,_enac_blocker",
        "CHEMBL.MECHANISM:hepatitis_c_virus_serine_protease,_ns3/ns4a_inhibitor",
        "CHEMBL.MECHANISM:1,3-beta-glucan_synthase_inhibitor",
        "CHEMBL.MECHANISM:3',5'-cyclic_phosphodiesterase_inhibitor",
        "CHEMBL.MECHANISM:dna_topoisomerase_i,_mitochondrial_inhibitor",
        "CHEMBL.MECHANISM:carbamoyl-phosphate_synthase_[ammonia],_mitochondrial_positive_allosteric_modulator",
        "CHEMBL.MECHANISM:parp_1,_2_and_3_inhibitor",
        "CHEMBL.MECHANISM:c-jun_n-terminal_kinase,_jnk_inhibitor",
        "CHEMBL.MECHANISM:voltage-gated_potassium_channel,_kqt;_kcnq2(kv7.2)/kcnq3(kv7.3)_activator",
        "CHEMBL.MECHANISM:hla_class_ii_histocompatibility_antigen,_drb1-10_beta_chain_other",
        "CHEMBL.MECHANISM:hla_class_ii_histocompatibility_antigen,_drb1-15_beta_chain_modulator",
        "CHEMBL.MECHANISM:indoleamine_2,3-dioxygenase_inhibitor",
        "CHEMBL.MECHANISM:5,6-dihydroxyindole-2-carboxylic_acid_oxidase_other",
        "CHEMBL.MECHANISM:amine_oxidase,_copper_containing_inhibitor",
        "CHEMBL.MECHANISM:carnitine_o-palmitoyltransferase_1,_muscle_isoform_inhibitor",
        "CHEMBL.MECHANISM:troponin,_cardiac_muscle_positive_modulator",
        "CHEMBL.MECHANISM:isocitrate_dehydrogenase_[nadp],_mitochondrial_inhibitor",
    ]

    t0 = timeit.default_timer()
    canonical_curies = synonymizer.get_canonical_curies(node_ids)
    t1 = timeit.default_timer()
    print(json.dumps(canonical_curies, sort_keys=True, indent=2))
    print("Elapsed time: " + str(t1 - t0))


# ############################################################################################
def run_example_11():
    synonymizer = NodeSynonymizer()

    print("==== Get equivalent curies for a set of input curies ============================")
    curies = ["DOID:14330", "UMLS:C0031485"]
    # curies = [ "DOID:14330", "UMLS:C0031485", "FMA:7203", "MESH:D005199", "CHEBI:5855", "DOID:9281xxxxx", "MONDO:0005520" ]

    t0 = timeit.default_timer()
    canonical_curies = synonymizer.get_equivalent_nodes(curies=curies, return_all_categories=True)
    t1 = timeit.default_timer()
    print(json.dumps(canonical_curies, sort_keys=True, indent=2))
    print("Elapsed time: " + str(t1 - t0))


# ############################################################################################
def run_example_12():
    synonymizer = NodeSynonymizer()

    print("==== Get full information in nouveau normalizer format  ============================")
    entities = ["DOID:14330", "anemia", "aardvark"]

    t0 = timeit.default_timer()
    normalizer_results = synonymizer.get_normalizer_results(entities=entities)
    t1 = timeit.default_timer()
    print(json.dumps(normalizer_results, sort_keys=True, indent=2))
    print("Elapsed time: " + str(t1 - t0))


# ############################################################################################
def run_examples():
    run_example_7()
    return
    run_example_1()
    run_example_2()
    run_example_3()
    run_example_4()
    run_example_5()
    run_example_6()
    run_example_7()
    run_example_8()
    run_example_9()
    run_example_10()


####################################################################################################
def main():
    import json

    parser = argparse.ArgumentParser(
        description="Tests or rebuilds the ARAX Node Synonymizer. Note that the build process requires 54 GB RAM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--build",
        action="store_true",
        help="If set, (re)build the index from scratch",
        default=False,
    )
    parser.add_argument(
        "-f",
        "--filter_file",
        action="store",
        help="When building, set an input filter_file to only build a testing subset, using file from previously used --export option",
        default=False,
    )
    parser.add_argument(
        "-s",
        "--save_state",
        action="store_true",
        help="If set, save the state of the build hashes when done reading source data (useful for subsequent --recollate)",
        default=False,
    )
    parser.add_argument(
        "-r",
        "--recollate",
        action="store_true",
        help="If set, try to load the previous saved state and recollate the nodes and write new tables",
        default=False,
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="If set, run a test of the index by doing several lookups",
        default=False,
    )
    parser.add_argument(
        "-l",
        "--lookup",
        action="store",
        help="If set to a curie or name, then use the NodeSynonymizer (or SRI normalizer) to lookup the equivalence information for the curie or name",
        default=None,
    )
    parser.add_argument(
        "-n",
        "--node_list",
        action="store",
        help="If set to a curie or name, then return a list of node curies that belong to the group for the specified curie or name",
        default=None,
    )
    parser.add_argument(
        "-e",
        "--export",
        action="store",
        help="Specify a filename that the lookup results will be exported to as json (e.g. curie.json)",
        default=None,
    )
    parser.add_argument(
        "-q",
        "--query",
        action="store_true",
        help="If set perform the test query and return",
        default=None,
    )
    parser.add_argument(
        "-g",
        "--get",
        action="store",
        help="Get nodes for the specified list in the specified kg_name",
        default=None,
    )
    parser.add_argument(
        "-c",
        "--live",
        action="store",
        help="Get the config.json field for the filename",
        default="Production",
    )
    parser.add_argument(
        "-u",
        "--update",
        action="store_true",
        help="If set, update the NodeSynonmizer with improved category information",
    )
    args = parser.parse_args()

    if (
        not args.build
        and not args.test
        and not args.recollate
        and not args.lookup
        and not args.node_list
        and not args.query
        and not args.get
        and not args.update
    ):
        parser.print_help()
        exit()

    synonymizer = NodeSynonymizer(live=args.live)

    # FW: If building use "node_synonymizer.sqlite" for the database name and reconnect
    if args.build or args.recollate:
        synonymizer.databaseName = "node_synonymizer.sqlite"
        synonymizer.connect()

    # If the user asks to perform the SELECT statement, do it
    if args.query:
        synonymizer.test_query()
        return

    # If the user asks to perform the SELECT statement, do it
    if args.update:
        synonymizer.update_categories()
        return

    # If the user asks to perform the SELECT statement, do it
    if args.get:
        t0 = timeit.default_timer()
        curies = args.get.split(",")
        results = synonymizer.get_equivalent_nodes(curies)
        t1 = timeit.default_timer()
        print(json.dumps(results, indent=2, sort_keys=True))
        print(f"INFO: Information retrieved in {t1 - t0} sec")
        return

    # If the --lookup option is provided, this takes precedence, perform the lookup and return
    if args.lookup is not None:
        t0 = timeit.default_timer()
        entities = args.lookup.split(",")
        equivalence = synonymizer.get_normalizer_results(entities)
        t1 = timeit.default_timer()
        print(json.dumps(equivalence, indent=2, sort_keys=True))
        if args.export:
            with open(args.export, "w") as outfile:
                outfile.write(json.dumps(equivalence, indent=2, sort_keys=True) + "\n")
        print(f"INFO: Information retrieved in {t1 - t0} sec")
        return

    # If the --node_list option is provided, just get the list of nodes that match and return that
    if args.node_list is not None:
        entities = args.node_list.split(",")
        equivalence = synonymizer.get_normalizer_results(entities)
        for entity in entities:
            if equivalence[entity] is not None:
                nodes = []
                for node in equivalence[entity]["nodes"]:
                    nodes.append(node["identifier"])
                print(f"{entity} = {nodes}")
        return

    # If the recollate option is selected, try to load the previous state
    if args.recollate:
        if not synonymizer.reload_state():
            return

    # Else if the build option is selected, build the kg_map from scratch
    elif args.build:
        print(
            "WARNING: Beginning full NodeSynonymizer build process. This requires 54 GB of RAM. If you don't have 54 GB of RAM available, this would be a good time to stop the process!"
        )
        synonymizer.build_kg_map(filter_file=args.filter_file)

        # print("WARNING: Skipping import_equivalenies because of possible problems")
        synonymizer.import_equivalencies()

        # print("WARNING: Skipping import_synonyms because of memory constraints")
        synonymizer.import_synonyms()

        # If the flag is set, save our state here for later recollate testing
        if args.save_state:
            synonymizer.save_state()

    # If either one is selected, do the collation and database writing
    if args.build or args.recollate:
        synonymizer.merge_unique_concepts()
        synonymizer.merge_unique_concepts_by_name()
        synonymizer.merge_unique_concepts()
        synonymizer.reprioritize_unique_concepts()

        synonymizer.create_tables()
        synonymizer.store_kg_map()
        synonymizer.create_indexes()

        print(
            f"INFO: Created a NodeSynonymizer database with\n  {len(synonymizer.kg_map['kg_nodes'])} nodes\n  {len(synonymizer.kg_map['kg_unique_concepts'])} unique concepts\n"
            + f"  {len(synonymizer.kg_map['kg_curies'])} curies\n  {len(synonymizer.kg_map['kg_names'])} names and abbreviations\n"
            + f"  {len(synonymizer.kg_map['kg_name_curies'])} name to curie provenance associations"
        )
        print("INFO: Processing complete. Freeing all the memory seems to still take a while though")

    # If requested, run the test examples
    if args.test:
        run_examples()


####################################################################################################
if __name__ == "__main__":
    main()
