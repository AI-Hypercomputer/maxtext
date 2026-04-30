<!--
# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
-->

# Contribute to documentation

The MaxText documentation website is built using [Sphinx](https://www.sphinx-doc.org)
and [MyST](https://myst-parser.readthedocs.io/en/latest/). Documents are written
in [MyST Markdown syntax](https://myst-parser.readthedocs.io/en/latest/syntax/typography.html#syntax-core).

## Building the documentation locally (optional)

If you are writing documentation for MaxText, you may want to preview the
documentation site locally to ensure things work as expected before a deployment
to [Read The Docs](https://about.readthedocs.com/?ref=app.readthedocs.org).

First, make sure you
[install MaxText from source](https://maxtext.readthedocs.io/en/latest/install_maxtext.html#from-source)
and install the necessary dependencies. You can do this by navigating to your
local clone of the MaxText repo and running:

```bash
uv pip install -r src/dependencies/requirements/requirements_docs.txt
```

Once the dependencies are installed and your `maxtext_venv` virtual environment
is activated, you can navigate to the `docs/` folder and run:

```bash
sphinx-build -b html . _build/html
```

This will generate the documentation in the `docs/_build/html` directory. These
files can be opened in a web browser directly, or you can use a simple HTTP
server to serve the files. For example, you can run:

```bash
python -m http.server -d _build/html
```

Then, open your web browser and navigate to `http://localhost:8000` to view the
documentation.

## Adding new documentation files

If you are adding a new document, make sure it is included in the `toctree`
directive corresponding to the section where the new document should live. For
example, if adding a new tutorial, make sure it is listed in
[the `docs/tutorials.md`](https://github.com/AI-Hypercomputer/maxtext/blob/7070e8eecbea8951c8e5281219ce797c8df1441f/docs/tutorials.md?plain=1#L38)
toctree.

## Documentation deployment

The latest version of the MaxText documentation, tracking the main branch of
development, is automatically deployed to
[https://maxtext.readthedocs.io/en/latest](https://maxtext.readthedocs.io/en/latest)
on any successful merge to the main branch.
