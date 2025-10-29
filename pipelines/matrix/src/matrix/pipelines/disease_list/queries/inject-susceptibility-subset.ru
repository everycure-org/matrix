PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX pattern: <http://purl.obolibrary.org/obo/mondo/patterns/>
PREFIX oboInOwl: <http://www.geneontology.org/formats/oboInOwl#>
PREFIX MONDO: <http://purl.obolibrary.org/obo/MONDO_>

INSERT {
    ?subset rdfs:subPropertyOf oboInOwl:SubsetProperty .
    ?entity oboInOwl:inSubset ?subset .
}

WHERE {
{
    ?entity rdfs:subClassOf+ MONDO:0042489 .
    BIND(IRI(CONCAT("http://purl.obolibrary.org/obo/mondo#","susceptibility_mondo")) AS ?subset)
} UNION {
    ?entity rdfs:subClassOf+ MONDO:0000001 .
    ?entity rdfs:label ?label .
    FILTER(regex(str(?label), "susceptib") || regex(str(?label), "predisposit") )
    BIND(IRI(CONCAT("http://purl.obolibrary.org/obo/mondo#","susceptibility_match")) AS ?subset)
}

}