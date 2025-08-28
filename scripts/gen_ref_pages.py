#!/usr/bin/env python

"""
This script auto-generates a single API reference page.

It iterates over all Python files in the `MaxText` directory,
and for each file, it adds a section to `reference.md` with
the correct mkdocstrings identifier.
"""
import ast
import os.path
from glob import glob

import mkdocs_gen_files

# The root of the package we want to document
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
src_root = os.path.join(REPO_ROOT, "MaxText")
docs_root = os.path.join(REPO_ROOT, "docs")
doc_file = os.path.join(docs_root, "reference.md")

orig_wd = os.getcwd()
if orig_wd != docs_root:
    os.chdir(REPO_ROOT)
try:
    with mkdocs_gen_files.open(doc_file, "wt", encoding="utf-8") as f:
        # Start with the main title
        f.write("# API Reference")

        # Find all python files
        for path in sorted(glob(os.path.join("**", "*.py"), root_dir=src_root, recursive=True)):
            # Create a module identifier from the path
            # e.g., MaxText/layers/attentions.py -> MaxText.layers.attentions
            rel_path = os.path.relpath(
                os.path.join(src_root, path), REPO_ROOT
            )
            identifier = rel_path.replace(os.path.sep, ".")[:-(len(os.path.extsep) + len("py"))]

            if os.path.basename(path) == "__init__.py":
                with open(os.path.join(src_root, path), "rt", encoding="utf-8") as f1:
                    if not ast.parse(f1.read()).body:
                        continue
                # Handle the top-level MaxText/__init__.py, which would otherwise be an empty string
                if not identifier:
                    identifier = "MaxText"

            # Add a markdown heading for each module
            f.write("\n---\n")
            f.write(f"## `{rel_path}`\n")
            # Add the mkdocstrings directive
            f.write(f"::: {identifier}")
        f.write("\n")

finally:
    if orig_wd != docs_root:
        os.chdir(orig_wd)
