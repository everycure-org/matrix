from matrix.pipelines.known_entity_removal.mondo_ontology import OntologyMONDO

# Constructing dummy overrides of OntologyMONDO to test the get_equivalent_mondo_ids method.


# "None" ontology override
class OwlOntologyNone:
    """A dummy class to represent a owlready2 ontology with no search results."""

    def search_one(self, id: str) -> None:
        return None


class OntologyMONDONone(OntologyMONDO):
    """A class overriding the owl ontology used in the OntologyMONDO class."""

    def __init__(self):
        self.ont = OwlOntologyNone()


# "Fixed ID" ontology override
class ThingClass:
    """A dummy class to represent a owlready2 thing class."""

    def __init__(self, id: str, include_is_a: bool = True):
        self.id = [id]
        if include_is_a:
            self.is_a = [ThingClass(id="descendant", include_is_a=False)]

    def subclasses(self):
        return [ThingClass(id="ancestor")]


class OwlOntologyTest:
    """A dummy class to represent a owlready2 ontology with search results."""

    def search_one(self, id: str) -> ThingClass:
        return ThingClass(id=id)


class OntologyMONDOTest(OntologyMONDO):
    """A class overriding the owl ontology used in the OntologyMONDO class."""

    def __init__(self):
        self.ont = OwlOntologyTest()


def test_none_ontology():
    # Given an ontology with no search results and any ID
    ontology = OntologyMONDONone()
    mondo_id = "mondo_id"
    # When we get equivalent Mondo IDs
    equivalent_mondo_ids = ontology.get_equivalent_mondo_ids(mondo_id)
    # Then we get an empty list
    assert equivalent_mondo_ids == []


def test_ontology_test():
    # Given an ontology with search results and any ID
    ontology = OntologyMONDOTest()
    mondo_id = "mondo_id"
    # When we get equivalent Mondo IDs
    equivalent_mondo_ids = ontology.get_equivalent_mondo_ids(mondo_id)
    # Then we get the expected list
    assert len(equivalent_mondo_ids) == 2
    assert set(equivalent_mondo_ids) == {"ancestor", "descendant"}
