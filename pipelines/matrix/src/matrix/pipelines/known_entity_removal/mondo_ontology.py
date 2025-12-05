import owlready2


class OntologyMONDO:
    """A class to get ancestor and descendent IDs from the MONDO ontology."""

    def __init__(self, owl_url: str = "https://purl.obolibrary.org/obo/mondo.owl"):
        """Initialize the ontology class with the given URL.

        Args:
            owl_url: A URL to download the MONDO ontology in OWL format.
        """
        self.ont = owlready2.get_ontology(owl_url).load()

    @staticmethod
    def _get_ids_from_owl_things(owl_things: list[owlready2.entity.ThingClass]) -> list[str]:
        return [thing.id[0] for thing in owl_things if hasattr(thing, "id")]

    def get_equivalent_mondo_ids(self, mondo_id: str) -> list[str]:
        mondo_class = self.ont.search_one(id=mondo_id)
        if mondo_class is None:
            return []
        ancestors = self._get_ids_from_owl_things(mondo_class.is_a)
        descendants = self._get_ids_from_owl_things(mondo_class.subclasses())
        return list(set(ancestors + descendants))


class OntologyTest(OntologyMONDO):
    """A class to override the OntologyMONDO class in the test environment.

    We're overriding the ontology class to avoid having to generate a consistent OWL file as part of or data fabrication process.
    """

    def __init__(self, **kwargs):
        pass

    def get_equivalent_mondo_ids(self, mondo_id: str) -> dict[str, list[str]]:
        """Return dummy IDs for testing."""
        return [mondo_id + "_ancestor", mondo_id + "_descendant"]
