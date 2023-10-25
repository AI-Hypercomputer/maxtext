export PYTHONPATH=$PWD
export XLMLTEST_CONFIGS=$PWD/configs/jsonnet/

find dags -name '*.py' |  xargs -n 1 -t python
