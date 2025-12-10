"""Pure python-based validator implementation."""

import csv
import os
import re
import sys
from collections import Counter, defaultdict

from tqdm import tqdm

from matrix_validator import util
from matrix_validator.validator import Validator

# Increase max field size
csv.field_size_limit(sys.maxsize)

#############################
# Configuration/Constants
#############################

# Required columns in edges file:
REQUIRED_EDGE_COLUMNS = [
    "subject",
    "predicate",
    "object",
    "primary_knowledge_source",
    "aggregator_knowledge_source",
    "knowledge_level",
    "agent_type",
]

# Required columns in nodes file:
REQUIRED_NODE_COLUMNS = ["id", "category"]

# A set of ALL valid KGX columns for edges (expand as needed)
VALID_KGX_EDGE_COLUMNS = {
    "publications",
    "publications_info",
    "kg2_ids",
    "qualified_predicate",
    "qualified_object_aspect",
    "qualified_object_direction",
    "domain_range_exclusion",
    "id",
    ":TYPE",
    ":START_ID",
    ":END_ID",
}.union(REQUIRED_EDGE_COLUMNS)

# A set of ALL valid KGX columns for nodes
VALID_KGX_NODE_COLUMNS = {
    "name",
    "all_names",
    "all_categories",
    "iri",
    "description",
    "equivalent_curies",
    "publications",
    ":LABEL",
}.union(REQUIRED_NODE_COLUMNS)

# Possible rule names that may appear in the violations dictionary.
# This is used to report "no violation" if they aren't triggered.
ALL_POSSIBLE_RULES = [
    "Naming convention violation",
    "Line endings violation",
    "Missing header row",
    "Empty header column name",
    "Duplicate header column name",
    "Invalid KGX column (but optional ones present)",
    "Missing required column",
    "Inconsistent columns",
    "Edge subject not found in node IDs",
    "Edge object not found in node IDs",
    "Leading or trailing whitespace",
    "Embedded TAB character",
    # etc., for any other rule names you have
]

BIOLINK_KNOWLEDGE_LEVEL_KEYS = util.get_biolink_model_knowledge_level_keys()
BIOLINK_AGENT_TYPE_KEYS = util.get_biolink_model_agent_type_keys()

# We store up to this many example lines for each violation
MAX_EXAMPLES_PER_VIOLATION = 5

#############################
# Utility Functions
#############################


def record_violation(violations_dict, rule_name, example):
    """
    Record a single violation for the specified rule.

    Keeps a few examples and increments a total count.
    """
    violations_dict[rule_name]["count"] += 1
    examples_list = violations_dict[rule_name]["examples"]
    if len(examples_list) < MAX_EXAMPLES_PER_VIOLATION:
        examples_list.append(example)


def check_file_naming_convention(file_path):
    """
    Check if the file follows the naming convention.

    For example:
        {kg_name}_{version}_edges.tsv   OR
        {kg_name}_{version}_nodes.tsv.
    """
    basename = os.path.basename(file_path)
    pattern = re.compile(r"^[a-z-_]+_[0-9a-z.-]+_(edges|nodes)\.tsv$")
    if not pattern.match(basename):
        return f"File '{basename}' does not match the expected pattern '{pattern.pattern}'"
    return None


def check_line_endings(file_path):
    r"""
    Check that file uses LF (\n) line endings, not CRLF (\r\n).

    We do a quick binary check for any '\r\n' sequence.
    """
    with open(file_path, "rb") as f:
        file_contents = f.read(1024 * 1024)  # Just check the first chunk.
        if b"\r\n" in file_contents:
            return "File uses CRLF (\\r\\n) instead of LF (\\n)"
    return None


def count_lines(file_path):
    """
    Count total number of lines in a file (for tqdm).

    For large files, consider alternative approaches if speed is an issue.
    """
    count = 0
    with open(file_path, "rb") as f:
        for _ in f:
            count += 1
    return count


def check_headers(header, violations_dict, file_type="edges"):
    """
    Check headers of table.

    In particular check:
      - No empty header cell values
      - No duplicate column names
      - Only valid KGX columns (others are "Invalid/optional KGX column")
      - All required columns are present.
    """
    # 1) No empty header cell
    for i, col in enumerate(header):
        if col.strip() == "":
            record_violation(violations_dict, "Empty header column name", f"Header column index {i} is empty.")

    # 2) No duplicates
    seen = set()
    for col in header:
        if col in seen:
            record_violation(violations_dict, "Duplicate header column name", f"Column '{col}' is duplicated.")
        else:
            seen.add(col)

    # 3) Only valid KGX columns
    if file_type == "edges":
        valid_cols = VALID_KGX_EDGE_COLUMNS
        required_cols = REQUIRED_EDGE_COLUMNS
    else:
        valid_cols = VALID_KGX_NODE_COLUMNS
        required_cols = REQUIRED_NODE_COLUMNS

    for col in header:
        if col not in valid_cols:
            record_violation(
                violations_dict,
                "Invalid/optional KGX column",
                f"Column '{col}' is not recognized as a KGX {file_type} column. (Might be optional)",
            )

    # 4) Check that all required columns are present
    for req in required_cols:
        if req not in header:
            record_violation(
                violations_dict,
                "Missing required column",
                f"Required column '{req}' is missing from the {file_type} file.",
            )


def check_row_structural(row, num_header_cols):
    """
    Return a string describing the violation if row length != num_header_cols.

    Else None if no violation.
    """
    if len(row) != num_header_cols:
        return f"Row has {len(row)} columns; expected {num_header_cols}. Data: {row}"
    return None


def check_cell_value(cell):
    """
    Check for embedded TAB, leading/trailing whitespace, etc.

    Return a list of violation descriptions (could be multiple).
    """
    issues = []
    if "\t" in cell:
        issues.append("Embedded TAB character")
    if cell != cell.strip():
        issues.append("Leading or trailing whitespace")
    return issues


def check_curie_prefix(curie, prefixes, found_unknown_prefixes):
    """
    Check if the CURIE's prefix is in the known BioLink prefixes.

    If not, store the unknown prefix in found_unknown_prefixes (a Counter or set).
    """
    if ":" not in curie:
        # No colon => invalid or no prefix
        found_unknown_prefixes["(no-colon)"] += 1
        return

    prefix = curie.split(":", 1)[0].strip()
    if prefix not in prefixes:
        found_unknown_prefixes[prefix] += 1


def load_node_ids(nodes_file, violations, prefixes, found_unknown_prefixes):
    """Validate the nodes file, gather node IDs, track unknown prefixes in found_unknown_prefixes."""
    error = check_file_naming_convention(nodes_file)
    if error:
        record_violation(violations, "Naming convention violation", error)

    # Check line endings
    error = check_line_endings(nodes_file)
    if error:
        record_violation(violations, "Line endings violation", f"nodes file: {error}")

    total_lines = count_lines(nodes_file)

    node_ids = set()
    with open(nodes_file, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = None
        num_header_cols = 0
        id_index = None

        for line_idx, row in enumerate(tqdm(reader, total=total_lines, desc="Reading nodes file")):
            if line_idx == 0:
                header = row
                if not header:
                    record_violation(violations, "Missing header row", "Nodes file has no columns in the header.")
                    break
                check_headers(header, violations, file_type="nodes")
                num_header_cols = len(header)
                try:
                    id_index = header.index("id")
                except ValueError:
                    id_index = None
                continue

            # Check structural row length
            structural_issue = check_row_structural(row, num_header_cols)
            if structural_issue:
                record_violation(violations, "Inconsistent columns", f"Line {line_idx + 1} (1-based) {structural_issue}")
                continue

            # Check cell-level
            for col_i, cell in enumerate(row):
                cell_issues = check_cell_value(cell)
                for issue in cell_issues:
                    record_violation(
                        violations,
                        issue,  # "Leading or trailing whitespace" or "Embedded TAB character"
                        f"Nodes file line {line_idx + 1}, column {header[col_i]!r} has issue: {issue}. Value: {cell}",
                    )

            # Collect node ID if present
            if id_index is not None and id_index < len(row):
                node_id = row[id_index]
                node_ids.add(node_id)
                check_curie_prefix(node_id, prefixes, found_unknown_prefixes)

    return node_ids


def check_edges_file(edges_file, node_ids, violations, prefixes, found_unknown_prefixes):
    """Validate the edges file."""
    error = check_file_naming_convention(edges_file)
    if error:
        record_violation(violations, "Naming convention violation", error)

    error = check_line_endings(edges_file)
    if error:
        record_violation(violations, "Line endings violation", f"edges file: {error}")

    total_lines = count_lines(edges_file)

    with open(edges_file, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = None
        num_header_cols = 0
        subj_idx = None
        obj_idx = None

        for line_idx, row in enumerate(tqdm(reader, total=total_lines, desc="Reading edges file")):
            if line_idx == 0:
                header = row
                if not header:
                    record_violation(violations, "Missing header row", "Edges file has no columns in the header.")
                    break
                check_headers(header, violations, file_type="edges")
                num_header_cols = len(header)
                try:
                    subj_idx = header.index("subject")
                except ValueError:
                    subj_idx = None
                try:
                    obj_idx = header.index("object")
                except ValueError:
                    obj_idx = None
                continue

            # Check structural row length
            structural_issue = check_row_structural(row, num_header_cols)
            if structural_issue:
                record_violation(violations, "Inconsistent columns", f"Line {line_idx + 1} {structural_issue}")
                continue

            # Cell-level checks
            for col_i, cell in enumerate(row):
                cell_issues = check_cell_value(cell)
                for issue in cell_issues:
                    record_violation(
                        violations,
                        issue,
                        f"Edges file line {line_idx + 1}, column {header[col_i]!r} has issue: {issue}. Value: {cell}",
                    )

            # Check subject/object membership + prefix
            if subj_idx is not None and subj_idx < len(row):
                subj = row[subj_idx]
                if subj not in node_ids:
                    record_violation(
                        violations,
                        "Edge subject not found in node IDs",
                        f"Line {line_idx + 1} subject '{subj}' not present in nodes file",
                    )
                check_curie_prefix(subj, prefixes, found_unknown_prefixes)

            if obj_idx is not None and obj_idx < len(row):
                obj = row[obj_idx]
                if obj not in node_ids:
                    record_violation(
                        violations,
                        "Edge object not found in node IDs",
                        f"Line {line_idx + 1} object '{obj}' not present in nodes file",
                    )
                check_curie_prefix(obj, prefixes, found_unknown_prefixes)


def report_violations(violations, found_unknown_prefixes):
    """
    Print out any rule violations, including "no violation found" messages.

    Also prints unknown prefixes (if any).
    """
    total_violations = sum(v["count"] for v in violations.values())
    # If there are no standard rule violations at all (empty dictionary),
    # we still want to see if we might have unknown prefixes.

    # 1) Print the "===== RULE VIOLATIONS REPORT ====="
    print("\n===== RULE VIOLATIONS REPORT =====")

    # 2) For each known possible rule, see if it's in the dictionary
    #    and whether it has nonzero count. If not present or zero, print no violation found.
    # something_reported = False
    for rule_name in ALL_POSSIBLE_RULES:
        if rule_name not in violations or violations[rule_name]["count"] == 0:
            print(f"  - No {rule_name} found.")
        else:
            # something_reported = True
            count = violations[rule_name]["count"]
            examples = violations[rule_name]["examples"]
            print(f"\n--- {rule_name} ---")
            print(f"Total: {count}")
            if examples:
                print("Examples:")
                for ex in examples:
                    print(f"  * {ex}")

    if total_violations == 0:
        # Overall, no standard violations were found
        print("\nGreat news! No standard rule violations were detected.")

    # 3) Now, handle unknown prefixes
    if len(found_unknown_prefixes) > 0:
        print("\n===== UNKNOWN PREFIXES FOUND =====")
        print("These prefixes are not recognized by the BioLink prefix list:")
        for prefix, count in found_unknown_prefixes.most_common():
            print(f"  {prefix} ({count} occurrences)")
    else:
        print("\nNo unknown prefixes found.")


class ValidatorPurePythonImpl(Validator):
    """Pure python-based validator implementation."""

    def __init__(self, config=None):
        """Create a new instance of the python-based validator."""
        super().__init__(config)

    def validate(self, nodes_file_path, edges_file_path, limit: int | None = None) -> int:
        """Validate a knowledge graph as nodes and edges KGX TSV files."""
        # Track all rule-based violations in this dictionary:
        #   { rule_name: { "count": int, "examples": [ ... ] }, ... }
        violations = defaultdict(lambda: {"count": 0, "examples": []})
        validation_reports = []

        # We'll store unknown prefixes in a Counter for multi-occurrence tallies
        found_unknown_prefixes = Counter()

        # 1) Load nodes
        node_ids = load_node_ids(nodes_file_path, violations, self.prefixes, found_unknown_prefixes)

        # 2) Check edges
        check_edges_file(edges_file_path, node_ids, violations, self.prefixes, found_unknown_prefixes)

        # 3) Generate validation report
        for rule_name in ALL_POSSIBLE_RULES:
            if rule_name in violations and violations[rule_name]["count"] > 0:
                count = violations[rule_name]["count"]
                examples = violations[rule_name]["examples"]
                report = f"{rule_name}: {count} occurrences\nExamples:\n"
                for ex in examples:
                    report += f"  * {ex}\n"
                validation_reports.append(report)

        # Report unknown prefixes
        if len(found_unknown_prefixes) > 0:
            report = "Unknown prefixes:\n"
            for prefix, count in found_unknown_prefixes.most_common():
                report += f"  {prefix} ({count} occurrences)\n"
            validation_reports.append(report)

        # Write validation report
        self.write_output(validation_reports)

        if len(validation_reports) > 0:
            return 1
        return 0
