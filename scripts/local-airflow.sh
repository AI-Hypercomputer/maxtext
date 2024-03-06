target_dag=$1

echo "Creating Airflow instance to run $target_dag..."

set -xue

# TODO(wcromar): enable editable installs of `xlml` instead
export PYTHONPATH=$PWD
export XLMLTEST_CONFIGS=$PWD/dags/jsonnet
export XLMLTEST_SSH_EXTERNAL_IPS=1

export AIRFLOW_HOME=$(mktemp -d)

mkdir $AIRFLOW_HOME/dags
ln -s $PWD/$target_dag $AIRFLOW_HOME/dags/

cd $AIRFLOW_HOME

export AIRFLOW__CORE__ALLOWED_DESERIALIZATION_CLASSES_REGEXP='.*'
airflow standalone
