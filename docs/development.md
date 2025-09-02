```{include} ../CONTRIBUTING.md
```

## Contributing documentation

### Building the documentation locally (optional)

If you are writing documentation for MaxText, you may want to preview the documentation site locally. First, make sure you install the necessary dependencies. You can do this by navigating to your local clone of the MaxText repo and running:
```bash
pip install -r requirements_docs.txt
```

Once the dependencies are installed, you can navigate to the `docs/` folder and run:

```bash
sphinx-build -b html . _build/html
```

This will generate the documentation in the `docs/_build/html` directory.

### Documentation deployment

The MaxText documentation is deployed to https://maxtext.readthedocs.io on any successful merge to the main branch.
