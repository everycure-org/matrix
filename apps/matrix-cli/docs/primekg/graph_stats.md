# Graph Statistics Report: **Graph**

This report provides a human-readable summary of the knowledge graph statistics for the PrimeKG transformation to KGX, including node distributions, edge counts, and schema details.

---

## 📊 High-Level Summary
| Metric | Value |
| :--- | :--- |
| **Total Nodes** | 134,437 |
| **Total Edges** | 8,765,586 |
| **Node Categories** | 10 |
| **Predicates** | 12 |

---

## 🧬 Node Statistics

### Nodes by Category
| Category | Count | Prefix(es) |
| :--- | :--- | :--- |
| `biolink:BiologicalProcess` | 28,642 | GO |
| `biolink:Gene` | 27,610 | NCBIGene |
| `biolink:Disease` | 22,205 | MONDO |
| `biolink:PhenotypicFeature` | 15,311 | HP |
| `biolink:GrossAnatomicalStructure` | 14,033 | UBERON |
| `biolink:MolecularActivity` | 11,169 | GO |
| `biolink:SmallMolecule` | 7,957 | DRUGBANK |
| `biolink:CellularComponent` | 4,176 | GO |
| `biolink:Pathway` | 2,516 | REACT |
| `biolink:ChemicalExposure` | 818 | CTD |
| `unknown` | 0 | - |

---

## 🔗 Edge Statistics

### Edges by Predicate
| Predicate | Count |
| :--- | :---: |
| `biolink:expressed_in` | 3,076,176 |
| `biolink:directly_physically_interacts_with` | 2,705,388 |
| `biolink:interacts_with` | 1,328,700 |
| `biolink:has_phenotype` | 1,128,226 |
| `biolink:associated_with` | 475,786 |
| `biolink:superclass_of` | 446,524 |
| `biolink:contraindicated_in` | 122,168 |
| `biolink:has_side_effect` | 129,568 |
| `biolink:treats` | 33,892 |
| `biolink:affected_by` | 18,528 |
| `biolink:correlated_with` | 12,054 |
| `biolink:applied_to_treat` | 9,500 |
| `unknown` | -720,924 |

### Detailed Edge Triplets (S-P-O)
This table breaks down counts by Subject, Predicate, and Object types.

| Subject Category | Predicate | Object Category | Count |
| :--- | :--- | :--- | :--- |
| SmallMolecule | `directly_physically_interacts_with` | SmallMolecule | 2,672,628 |
| Gene | `expressed_in` | GrossAnatomicalStructure | 1,538,088 |
| GrossAnatomicalStructure | `expressed_in` | Gene | 1,538,088 |
| Gene | `interacts_with` | Gene | 642,150 |
| Disease | `has_phenotype` | PhenotypicFeature | 564,113 |
| PhenotypicFeature | `has_phenotype` | Disease | 564,113 |
| Disease | `associated_with` | Gene | 234,563 |
| Gene | `associated_with` | Disease | 234,563 |
| Disease | `superclass_of` | Disease | 229,168 |
| PhenotypicFeature | `has_side_effect` | SmallMolecule | 64,784 |
| SmallMolecule | `has_side_effect` | PhenotypicFeature | 64,784 |
| Disease | `contraindicated_in` | SmallMolecule | 61,084 |
| SmallMolecule | `contraindicated_in` | Disease | 61,084 |
| Pathway | `interacts_with` | Gene | 42,646 |
| Gene | `interacts_with` | Pathway | 42,646 |
| PhenotypicFeature | `superclass_of` | PhenotypicFeature | 37,472 |
| GrossAnatomicalStructure | `superclass_of` | GrossAnatomicalStructure | 28,064 |
| Disease | `treats` | SmallMolecule | 16,946 |
| SmallMolecule | `treats` | Disease | 16,946 |
| Gene | `directly_physically_interacts_with` | SmallMolecule | 16,380 |
| SmallMolecule | `directly_physically_interacts_with` | Gene | 16,380 |
| BiologicalProcess | `superclass_of` | BiologicalProcess | 13,464 |
| BiologicalProcess | `interacts_with` | Gene | 10,263 |
| Gene | `interacts_with` | BiologicalProcess | 10,263 |
| Gene | `affected_by` | SmallMolecule | 9,264 |
| SmallMolecule | `affected_by` | Gene | 9,264 |
| Pathway | `superclass_of` | Pathway | 5,070 |
| Disease | `applied_to_treat` | SmallMolecule | 4,750 |
| SmallMolecule | `applied_to_treat` | Disease | 4,750 |
| Gene | `associated_with` | PhenotypicFeature | 3,330 |
| PhenotypicFeature | `associated_with` | Gene | 3,330 |
| Gene | `interacts_with` | MolecularActivity | 1,439 |
| MolecularActivity | `interacts_with` | Gene | 1,439 |
| CellularComponent | `interacts_with` | Gene | 948 |
| Gene | `interacts_with` | CellularComponent | 948 |
| CellularComponent | `superclass_of` | CellularComponent | 304 |
| MolecularActivity | `superclass_of` | MolecularActivity | 70 |