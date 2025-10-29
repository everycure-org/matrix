PREFIX mondo: <http://purl.obolibrary.org/obo/mondo#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX oio: <http://www.geneontology.org/formats/oboInOwl#>

INSERT {
  ?descendant oio:inSubset ?memberSubset .
  ?descendant oio:inSubset ?subjectSubset .
}
WHERE {
    VALUES ?subset { 
      <http://purl.obolibrary.org/obo/mondo#mondo_txgnn> 
      <http://purl.obolibrary.org/obo/mondo#harrisons_view>
      <http://purl.obolibrary.org/obo/mondo#mondo_top_grouping> 
      }
    ?subject oio:inSubset ?subset ; rdfs:label ?label .
    ?descendant rdfs:subClassOf+ ?subject .
    BIND(IRI(CONCAT(STR(?subset),"_member")) AS ?memberSubset)
    BIND(IRI(CONCAT(CONCAT(STR(?subset),"_"),REPLACE(LCASE(?label), "[^a-zA-Z0-9]", "_"))) AS ?subjectSubset)
}
