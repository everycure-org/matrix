PREFIX mondo: <http://purl.obolibrary.org/obo/mondo#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX oio: <http://www.geneontology.org/formats/oboInOwl#>

INSERT {
  ?subject oio:inSubset <http://purl.obolibrary.org/obo/mondo#mondo_top_grouping> .
}
WHERE {
  ?subject rdfs:subClassOf <http://purl.obolibrary.org/obo/MONDO_0700096> .
}
