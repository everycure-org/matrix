#!/usr/bin/python3
## This code is created by RTX team (https://github.com/RTXteam/RTX/) used in the version 2.7.3 of RTX-KG2c (Wood et al. 2021 doi: 10.1101/2021.10.17.464747)
"""A class to access BioLink categories"""

import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


import copy
import json
import os
import pickle

import requests
import requests_cache


# Class that provides a simple interface to BioLink categories and their ancestors and approved conflations
class CategoryManager:
    # Constructor
    def __init__(self):
        self.location = os.path.dirname(os.path.abspath(__file__))
        requests_cache.install_cache(self.location + "/category_manager.cache")

        self.categories = {"ancestors": {}, "relevant_categories": {}}

        self.approved_conflations = {
            "biolink:Gene": ["biolink:Protein"],
            "biolink:Protein": ["biolink:Gene"],
            # Decided 2021-07-28 mini-hackathon that we may be best off NOT doing conflation here. Just use
            # ChemicalEntity to refer to everything
            # #'biolink:Drug': [ 'biolink:ChemicalEntity', 'biolink:MolecularEntity', 'biolink:SmallMolecule' ],
            #'biolink:ChemicalEntity': [ 'biolink:Drug', 'biolink:MolecularEntity', 'biolink:SmallMolecule' ],
            #'biolink:SmallMolecule': [ 'biolink:Drug', 'biolink:MolecularEntity', 'biolink:ChemicalEntity' ],
            #'biolink:MolecularEntity': [ 'biolink:Drug', 'biolink:SmallMolecule', 'biolink:ChemicalEntity' ],
            "biolink:Disease": ["biolink:PhenotypicFeature"],
            "biolink:PhenotypicFeature": ["biolink:Disease"],
            "biolink:DiseaseOrPhenotypicFeature": [
                "biolink:Disease",
                "biolink:PhenotypicFeature",
            ],
        }

    # ############################################################################################
    # Store the cache of all category information
    # Not actually used. Rely on web cache for now
    def store_cache(self):
        if self.categories is None:
            return
        filename = self.location + "/category_manager.pickle"
        print(f"INFO: Storing category_manager cache to {filename}")
        with open(filename, "wb") as outfile:
            pickle.dump(self.categories, outfile)

    # ############################################################################################
    # Load the cache of all category information
    # Not actually used. Rely on web cache for now
    def load_cache(self):
        filename = self.location + "/category_manager.pickle"
        if os.path.exists(filename):
            print(f"INFO: Reading category_manager cache from {filename}")
            with open(filename, "rb") as infile:
                self.categories = pickle.load(infile)
        else:
            print(f"INFO: category_manager cache {filename} does not yet exist. Need to create it first.")

    # ############################################################################################
    # Retrieve the ancestors of a biolink category from SRI web service
    def get_ancestors(self, category):
        if category is None:
            return

        if not isinstance(category, str):
            raise (f"ERROR: category {category} is not type str")

        # If we alreacy have the information, return the cached result
        if category in self.categories["ancestors"]:
            return self.categories["ancestors"][category]

        # Build the URL and fetch the result
        url = f"https://bl-lookup-sri.renci.org/bl/{category}/ancestors?version=2.1.0"
        response_content = requests.get(url, headers={"accept": "application/json"})
        status_code = response_content.status_code

        # Check for a returned error
        if status_code != 200:
            eprint(f"WARNING: returned with status {status_code} while retrieving ancestors for {category}")
            response_list = [category]
            self.categories["ancestors"][category] = response_list
            return response_list

        # Unpack the response
        response_list = response_content.json()

        self.categories["ancestors"][category] = response_list

        return response_list

    # ############################################################################################
    # Retrieve the ancestors of a biolink category
    def get_expansive_categories(self, categories):
        # If no categories are provided, then there's nothing to do
        if categories is None:
            return

        # If the supplied categories is just a string, then turn is into the expected list
        if isinstance(categories, str):
            categories = [categories]

        # If the supplied categories is not a list, then error out
        if not isinstance(categories, list):
            raise (f"ERROR: categories {categories} must be type list or str")

        # Create a dict to store of the computed expansive categories
        expansive_categories = {}

        # Loop over all the supplied categories and merge them into a single list
        for category in categories:
            # First add the category itself plus any approved conflations
            expansive_categories[category] = True
            if category in self.approved_conflations:
                for conflated_category in self.approved_conflations[category]:
                    expansive_categories[conflated_category] = True

            parent_expansive_categories = copy.copy(expansive_categories)
            for parent_category in parent_expansive_categories:
                # Get the ancestors of this category
                ancestors = self.get_ancestors(parent_category)

                # If we didn't get any, that's fine, just move on
                if ancestors is None:
                    continue

                # For each discovered ancestor, add it to the dict and also add any approved conflations
                for ancestor in ancestors:
                    if ancestor == "biolink:Entity":
                        continue
                    expansive_categories[ancestor] = True
                    if ancestor in self.approved_conflations:
                        for conflated_category in self.approved_conflations[ancestor]:
                            expansive_categories[conflated_category] = True

        return expansive_categories


# ############################################################################################
# Command line interface for this class
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Interface to the Category Manager",
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
        "-a",
        "--ancestors",
        action="store",
        help="Look up the ancestors for the supplied category",
        default=False,
    )
    parser.add_argument(
        "-c",
        "--categories",
        action="store",
        help="Get the full list of exansive categories for the supplied category",
        default=False,
    )
    args = parser.parse_args()

    if not args.build and not args.ancestors and not args.categories:
        parser.print_help()
        sys.exit(2)

    catman = CategoryManager()

    if args.ancestors:
        ancestors = catman.get_ancestors(args.ancestors)
        print(json.dumps(ancestors, indent=2, sort_keys=True))
        return

    if args.categories:
        categories = catman.get_expansive_categories(args.categories)
        print(json.dumps(categories, indent=2, sort_keys=True))
        return


if __name__ == "__main__":
    main()
