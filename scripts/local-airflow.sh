target_dag=$1

echo "Creating Airflow instance to run $target_dag..."

set -xue

# TODO(wcromar): enable editable installs of `xlml` instead
export PYTHONPATH=$PWD
export XLMLTEST_CONFIGS=$PWD/dags/jsonnet
export XLMLTEST_MULTIPOD_LEGACY_TEST_DIR=$PWD/dags/multipod/legacy_tests
export XLMLTEST_SSH_EXTERNAL_IPS=1
export XLMLTEST_LOCAL_AIRFLOW=1

export AIRFLOW_HOME=$(mktemp -d)

mkdir $AIRFLOW_HOME/dags
ln -s $PWD/$target_dag $AIRFLOW_HOME/dags/

cd $AIRFLOW_HOME

# Disable auth login
echo "AUTH_ROLE_PUBLIC = 'Admin'" > webserver_config.py

# Configs from https://airflow.apache.org/docs/apache-airflow/stable/configurations-ref.html
export AIRFLOW__CORE__ALLOWED_DESERIALIZATION_CLASSES_REGEXP='.*'
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__WEBSERVER__WEB_SERVER_HOST=localhost
export AIRFLOW__API__AUTH_BACKENDS=airflow.api.auth.backend.default
airflow standalone
