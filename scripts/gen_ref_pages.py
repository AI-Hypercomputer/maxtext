"""
This script auto-generates a single API reference page.

It iterates over all Python files in the `MaxText` directory,
and for each file, it adds a section to `reference.md` with
the correct mkdocstrings identifier.
"""
from pathlib import Path
import mkdocs_gen_files

# The root of the package we want to document
src_root = Path("MaxText")
doc_file = "reference.md"

with mkdocs_gen_files.open(doc_file, "w") as f:
    # Start with the main title
    print("# API Reference", file=f)

    # Find all python files
    for path in sorted(src_root.rglob("*.py")):
        # Create a module identifier from the path
        # e.g., MaxText/layers/attentions.py -> MaxText.layers.attentions
        module_path = path.relative_to('.').with_suffix('')
        identifier = ".".join(module_path.parts)

        # This block is the crucial fix:
        if path.name == "__init__.py":
            # Skip empty __init__ files, as they have no content to document
            if not path.read_text(encoding="utf-8").strip():
                continue
            # The identifier for an __init__.py is its parent directory (the package name)
            identifier = ".".join(module_path.parts[:-1])
            # Handle the top-level MaxText/__init__.py, which would otherwise be an empty string
            if not identifier:
                identifier = "MaxText"

        # Add a markdown heading for each module
        print(f"\n---\n", file=f)
        print(f"## `{identifier}`\n", file=f)
        # Add the mkdocstrings directive
        print(f"::: {identifier}", file=f)
