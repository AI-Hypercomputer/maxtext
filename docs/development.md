```{include} ../CONTRIBUTING.md
```

## Contributing documentation

The MaxText documentation website is built using [Sphinx](https://www.sphinx-doc.org) and [MyST](https://myst-parser.readthedocs.io/en/latest/). Documents are written in [MyST Markdown syntax](https://myst-parser.readthedocs.io/en/latest/syntax/typography.html#syntax-core).

### Building the documentation locally (optional)

If you are writing documentation for MaxText, you may want to preview the documentation site locally to ensure things work as expected before a deployment to Read The Docs.

First, make sure you install the necessary dependencies. You can do this by navigating to your local clone of the MaxText repo and running:

```bash
pip install -r requirements_docs.txt
```

Once the dependencies are installed, you can navigate to the `docs/` folder and run:

```bash
sphinx-build -b html . _build/html
```

This will generate the documentation in the `docs/_build/html` directory. These files can be opened in a web browser directly, or you can use a simple HTTP server to serve the files. For example, you can run:

```bash
python -m http.server -d docs/_build/html
```

Then, open your web browser and navigate to `http://localhost:8000` to view the documentation.

### Adding new documentation files

If you are adding a new document, make sure it is included in the `toctree` directive corresponding to the section where the new document should live. For example, if adding a new tutorial, make sure it is listed in [the `docs/tutorials.md`](https://github.com/AI-Hypercomputer/maxtext/blob/7070e8eecbea8951c8e5281219ce797c8df1441f/docs/tutorials.md?plain=1#L38).

### Documentation deployment

The MaxText documentation is deployed to [https://maxtext.readthedocs.io](https://maxtext.readthedocs.io) on any successful merge to the main branch.
