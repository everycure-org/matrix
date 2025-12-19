"""matrix-validator checks package."""

NCNAME_PATTERN = r"([A-Za-z_][A-Za-z0-9\.\-_]+):"
LOCAL_UNIQUE_IDENTIFIER_PATTERN = r"(/[^\s/][^\s]*|[^\s/][^\s]*|[^\s]?)"

CURIE_REGEX = rf"^{NCNAME_PATTERN}{LOCAL_UNIQUE_IDENTIFIER_PATTERN}$"
STARTS_WITH_BIOLINK_REGEX = rf"^biolink:{LOCAL_UNIQUE_IDENTIFIER_PATTERN}$"
DELIMITED_BY_PIPES = r"^[^,|]*(\|[^,|]*)*$"
NO_LEADING_WHITESPACE = r"^[^ ].+$"
NO_TRAILING_WHITESPACE = r"^.+[^ ]$"
