from matrix.pipelines.known_entity_removal.mondo_ontology import OntologyMONDO


class OwlOntologyNone:
    def search_one(self, id: str) -> None:
        return None


class OntologyMONDONone(OntologyMONDO):
    def __init__(self):
        self.ont = OwlOntologyNone()


class ThingClass:
    def __init__(self, id: str):
        self.id = [id]


class MONDOClass:
    def __init__(self):
        self.is_a = [ThingClass(id="descendant")]

    def subclasses(self):
        return [ThingClass(id="ancestor")]


class OwlOntologyTest:
    def search_one(self, id: str) -> MONDOClass:
        return MONDOClass()


class OntologyMONDOTest(OntologyMONDO):
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
    assert equivalent_mondo_ids == ["ancestor", "descendant"]
