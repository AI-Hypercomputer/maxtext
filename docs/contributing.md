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

We use the code formatter [Pyink](https://github.com/google/pyink), which is a fork of the [Black](https://github.com/psf/black) formatter with a few different behaviors tailored towards Google internal repository.

#### Step 1: Install Pyink.

*For Googlers:*

Run `sudo apt install pipx; pipx install pyink` on your Cloudtop to install Pyink.

*For external contributors:*

Run `pip install pyink` on your machine to install Pyink.

#### Step 2: Format codes.

Run `bash scripts/format-codes.sh` to format your codes.

### Code Reviews

All submissions, including submissions by project members, require review. We
use [GitHub pull requests](https://docs.github.com/articles/about-pull-requests)
for this purpose.

### Testing Changes Locally

To run a standalone Airflow server locally without uploading your changes, run the following from your copy of this repository:

```
gcloud auth login --update-adc
AIRFLOW_HOME=$PWD PYTHONPATH=. airflow standalone
```

This functionality is extremely experimental, and not all DAGs are expected to work with a local standalone server. Only the Airflow server runs locally. Tests will still run in the project defined in each DAG, so use this option with caution.
