import json

cm = json.load(open("codemeta.json"))

schema = {
  "@context": "https://schema.org",
  "@type": "SoftwareSourceCode",
  "name": cm.get("name"),
  "description": cm.get("description"),
  "url": cm.get("url"),
  "codeRepository": cm.get("codeRepository"),
  "programmingLanguage": cm.get("programmingLanguage"),
  "license": cm.get("license"),
  "version": cm.get("version"),
  "author": cm.get("author"),
  "datePublished": cm.get("datePublished"),
  "keywords": cm.get("keywords"),
  "identifier": {
    "@type": "PropertyValue", "propertyID": "DOI", "value": cm.get("identifier")
  } if isinstance(cm.get("identifier"), str) else None
}
schema = {k:v for k,v in schema.items() if v}
json.dump(schema, open("schemaorg.jsonld","w"), indent=2)
