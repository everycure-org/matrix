from dataclasses import dataclass


class OwlOntologyNone:
    def search_one(self, id: str) -> None:
        return None


class ThingClass:
    def __init__(self, id: str):
        self.id = [id]


@dataclass
class MONDOClass:
    is_a: list[ThingClass] = [ThingClass(id="descendant")]

    def subclasses(self):
        return [ThingClass(id="ancestor")]


class OwlOntologyTest:
    def search_one(self, id: str) -> MONDOClass:
        return MONDOClass()


def test_none_ontology():
    # Given an ontology with no search results and any ID
    ontology = OwlOntologyNone
    mondo_id = "mondo_id"
    # When we get
    equivalent_mondo_ids = ontology.get_equivalent_mondo_ids("MONDO:0000001")
    # Then we should get an empty list
    assert equivalent_mondo_ids == []
