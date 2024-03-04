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

### Code Style

We use the code linter [Pylint](https://github.com/pylint-dev/pylint) and formatter [Pyink](https://github.com/google/pyink).
* Pylint, a widely-used static code analyzer, works well with Google's internal code standards.
* Pyink is a fork of the [Black](https://github.com/psf/black) formatter with a few different behaviors tailored towards Google internal repository.

#### Step 1: Install Pylint and Pyink.

*For Googlers:*

Run `sudo apt install pipx; pipx install pylint==3.1.0 --force; pipx install pyink==23.10.0` on your Cloudtop to install Pylint and Pyink.

*For external contributors:*

Run `pip install pylint==3.1.0; pip install pyink==23.10.0` on your machine to install Pylint and Pyink.

#### Step 2: Clean up codes.

Run `bash scripts/code-style.sh` to clean up your codes. You will need to see `Successfully clean up all codes` in the output to avoid check failures in your PR.

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
