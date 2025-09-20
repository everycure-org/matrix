import json
import sys

import yaml

cm = json.load(open("codemeta.json"))
def person(p):
    out = {"name": p.get("name")}
    aff = p.get("affiliation", {})
    if isinstance(aff, dict) and aff.get("name"):
        out["affiliation"] = aff["name"]
    return out

cff = {
  "cff-version": "1.2.0",
  "message": "If you use this software, please cite it as below.",
  "title": cm.get("name"),
  "abstract": cm.get("description"),
  "version": cm.get("version"),
  "repository-code": cm.get("codeRepository") or cm.get("url"),
  "license": (cm.get("license") or "").split("/")[-1] or None,
  "doi": cm.get("identifier") if isinstance(cm.get("identifier"), str) else None,
  "date-released": cm.get("datePublished"),
  "authors": [person(a) for a in cm.get("author", [])]
}
# drop None
cff = {k:v for k,v in cff.items() if v}
yaml.safe_dump(cff, open("CITATION.cff","w"), sort_keys=False)
