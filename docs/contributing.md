# How to Contribute

We would love to accept your patches and contributions to this project.

## Before you begin

#### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

#### Review our Community Guidelines

This project follows [Google's Open Source Community
Guidelines](https://opensource.google/conduct/).

## Contribution process

#### Setting up development environment with VSCode

Most of the team uses Visual Studio Code with the [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python) for development. We recommend you install the [workspace recommended extensions](https://code.visualstudio.com/docs/editor/extension-marketplace#_workspace-recommended-extensions) with the `Extensions: Show Recommended Extensions` command. Repository defaults for linting and formatting are in `.vscode/settings.json`.

To create a local virtual environment for development, use the [`Python: Create Environment`](https://code.visualstudio.com/docs/python/environments#_using-the-create-environment-command). Remember to select both `dev-requirements.txt` and `requirements.txt` when prompted to install dependencies. Alternatively, you can run `venv` from the command line and install the requirements yourself with `pip install -r`.

Once you've set up your virtual environment, ensure you are using the correct intepreter from your local `.venv` directory from the [`Python: Select Interpreter`](https://code.visualstudio.com/docs/python/environments#_select-and-activate-an-environment) command.

When we update the development requirements (especially Airflow itself), you may want to just delete your `.venv` directory and run through these instructions again.

#### Code style

We use the code linter [Pylint](https://github.com/pylint-dev/pylint) and formatter [Pyink](https://github.com/google/pyink).

* Pylint, a widely-used static code analyzer, works well with Google's internal code standards.
* Pyink is a fork of the [Black](https://github.com/psf/black) formatter with a few different behaviors tailored towards Google internal repository.

You have two ways:
* Format an [individual Python file through VSCode](https://code.visualstudio.com/docs/python/formatting#_format-your-code).
* Run formatting and linting for the whole repository, run `bash scripts/code-style.sh`. You will need to see `Successfully clean up all codes` in the output to avoid check failures in your PR.

#### JSonnet (optional)

JSonnet is only required for local testing for some tests, primarily PyTorch/XLA's. Install the latest version of [`go-jsonnet`](https://github.com/google/go-jsonnet) to be able to generate test configs locally.

#### Testing changes locally

To run a dag file in a temporary local environment, use `local-airflow.sh`. The script will symlink just the DAG provided to speed up parsing times.

This requires Airflow to be installed locally. You can configure your local environment by running `pip install -r .github/requirements.txt`.

To run the local environment, use the following commands:

```
gcloud auth login --update-adc
scripts/local-airflow.sh path/to/dag_file.py
```

Comment out any test cases in the DAG that you do not want to run, or create a temporary DAG file to avoid running all tests.


##### XPK-based tests

XPK will run in the same environment as the local airflow execution, and there are two XPK requirements to be aware of:

1. Python version >= 3.10.
1. kubectl must support GKE-based authentication. Install the required component using

```
gcloud components install gke-gcloud-auth-plugin
```

If you encounter an error related to the gcloud installation being `managed by an external package manager`, you'll need to reinstall. The simplest way is to follow https://cloud.google.com/sdk/docs/downloads-interactive.


##### JSonnet-based tests

If you're running a JSonnet-based test, run this each time any time the test changes:

```
scripts/gen-configs.sh
```

Airflow will print a link to a local instance. From the UI, find your dag and run it manually.

This functionality is extremely experimental, and not all DAGs are expected to work with a local standalone server. Only the Airflow server runs locally. Tests will still run in the project defined in each DAG, so use this option with caution.

#### Code reviews

All submissions, including submissions by project members, require review. We use [GitHub pull requests](https://docs.github.com/articles/about-pull-requests) for this purpose.
