from pathlib import Path

import mkdocs_gen_files

src_root = Path("../pipelines/matrix/src/")
for path in src_root.glob("**/*.py"):
    doc_path = Path("reference", path.relative_to(src_root)).with_suffix(".md")
    module_path = Path(path.relative_to(src_root)).with_suffix(".py")

    # ignore __init__.py files
    parts = module_path.with_suffix("").parts
    if "__init__" == parts[-1]:
        parts = parts[:-1]

    # import os
    # p = Path("/tmp/") / doc_path
    # os.makedirs(p.parent, exist_ok=True)
    # with open(Path("/tmp/") / doc_path, "w") as f:
    with mkdocs_gen_files.open(doc_path, "w") as f:
        # create python module structure
        ident = ".".join(parts)
        # print the module structure to generate docstrings
        print("::: " + ident, file=f)
