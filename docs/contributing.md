# How to Contribute

We would love to accept your patches and contributions to this project.

## Before you begin

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our Community Guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google/conduct/).

## Contribution process

## Setting up development environment with VSCode

Most of the team uses Visual Studio Code with the [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) for development. We recommend you install the [workspace recommended extensions](https://code.visualstudio.com/docs/editor/extension-marketplace#_workspace-recommended-extensions) with the `Extensions: Show Recommended Extensions` command. Repository defaults for linting and formatting are in `.vscode/settings.json`.

To create a local virtual environment for development, use the [`Python: Create Environment`](https://code.visualstudio.com/docs/python/environments#_using-the-create-environment-command). Remember to select both `dev-requirements.txt` and `requirements.txt` when prompted to install dependencies. Alternatively, you can run `venv` from the command line and install the requirements yourself with `pip install -r`.

Once you've set up your virtual environment, ensure you are using the correct intepreter from your local `.venv` directory from the [`Python: Select Interpreter`](https://code.visualstudio.com/docs/python/environments#_select-and-activate-an-environment) command.

### Code Style

We use the code linter [Pylint](https://github.com/pylint-dev/pylint) and formatter [Pyink](https://github.com/google/pyink).

* Pylint, a widely-used static code analyzer, works well with Google's internal code standards.
* Pyink is a fork of the [Black](https://github.com/psf/black) formatter with a few different behaviors tailored towards Google internal repository.

You can format an [individual Python file through VSCode](https://code.visualstudio.com/docs/python/formatting#_format-your-code). To run formatting and linting for the whole repository, run `bash scripts/code-style.sh`. You will need to see `Successfully clean up all codes` in the output to avoid check failures in your PR.

### Code Reviews

All submissions, including submissions by project members, require review. We
use [GitHub pull requests](https://docs.github.com/articles/about-pull-requests)
for this purpose.

### Testing Changes Locally

To run a dag file in a temporary local environment, use the `local-airflow.sh`:

```
gcloud auth login --update-adc
scripts/local-airflow.sh path/to/dag_file.py
```

The script will symlink just the DAG provided to speed up parsing times.

This functionality is extremely experimental, and not all DAGs are expected to work with a local standalone server. Only the Airflow server runs locally. Tests will still run in the project defined in each DAG, so use this option with caution.
