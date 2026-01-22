import importlib.resources
from typing import Dict

from linkml.utils.schema_builder import SchemaBuilder
from linkml_runtime import SchemaView
from linkml_runtime.dumpers import yaml_dumper
from linkml_runtime.linkml_model import EnumDefinition, PermissibleValue
from linkml_runtime.utils.formatutils import camelcase, underscore

PREDICATE_ROOT = "related to"
NODE_CATEGORY_ROOT = "named thing"
BIOLINK_IMPORT_SCHEMA_FILE = "biolink_imports.yaml"


class FlatteningExtractor:
    def __init__(self, sv: SchemaView):
        self.sv = sv

    def generate_enum(self, name: str, title_to_value_mapping: Dict[str, str]) -> EnumDefinition:
        permissible_values = [
            PermissibleValue(title=title, text=value) for title, value in title_to_value_mapping.items()
        ]
        enum = EnumDefinition(name=name, permissible_values=permissible_values)
        return enum

    def generate_enum_from_slot_descendants(self, name: str, root_slot: str) -> EnumDefinition:
        values = {}
        for slot in self.sv.slot_descendants(root_slot):
            title = underscore(slot).upper()
            value = self.sv.schema.default_prefix + ":" + underscore(slot)
            values[title] = value
        return self.generate_enum(name, values)

    def generate_enum_from_class_descendants(self, name: str, root_class: str) -> EnumDefinition:
        values = {}
        for cls in self.sv.class_descendants(root_class):
            title = underscore(cls).upper()
            value = self.sv.schema.default_prefix + ":" + camelcase(cls)
            values[title] = value
        return self.generate_enum(name, values)


def get_biolink_model_schemaview() -> SchemaView:
    # Locate the YAML file for the biolink model within the biolink_model package
    biolink_model_file = importlib.resources.files("biolink_model.schema").joinpath("biolink_model.yaml")
    return SchemaView(str(biolink_model_file))


def generate_biolink_enum_schema():
    sv = get_biolink_model_schemaview()
    fe = FlatteningExtractor(sv)
    # Create the schema builder
    sb = SchemaBuilder("biolink_enum")
    sb.schema.default_prefix = "matrix_schema"
    sb.schema.id = "https://w3id.org/everycure-org/matrix-schema/biolink_imports"
    sb.schema.description = "Biolink model enums for the matrix schema"
    sb.schema.license = "BSD-3"

    # Add all predicate values as an enum for the predicate slot in the matrix schema
    sb.add_enum(fe.generate_enum_from_slot_descendants(name="PredicateEnum", root_slot=PREDICATE_ROOT))

    # Add all node category values as an enum for the node category slot in the matrix schema
    sb.add_enum(fe.generate_enum_from_class_descendants(name="NodeCategoryEnum", root_class=NODE_CATEGORY_ROOT))

    # Additionally, bring over KL & AT enums from the biolink model, stripping subsets that aren't defined here
    knowledge_level_enum = sv.get_enum("KnowledgeLevelEnum")
    knowledge_level_enum.in_subset = []
    sb.add_enum(knowledge_level_enum)

    agent_type_enum = sv.get_enum("AgentTypeEnum")
    agent_type_enum.in_subset = []
    sb.add_enum(agent_type_enum)

    output_path = f"src/matrix_validator/resources/{BIOLINK_IMPORT_SCHEMA_FILE}"
    with open(output_path, "w") as f:
        f.write(yaml_dumper.dumps(sb.schema))


generate_biolink_enum_schema()
