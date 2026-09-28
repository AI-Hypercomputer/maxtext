# test instruction

```
cd ~/maxtext
activate ~/venv-maxtext
```

1.
use shared pathways service (SPS) for v7x-8, run this unit test
```
python3 -m pytest -v --pyargs tests.unit.moe_test -k "test_moe_quantize_combine_bwd_method" -rP -s
```

2.
```
export PYTHONPATH="$PWD/src"
python3 -m pytest -v --pyargs tests.unit.moe_test -k "test_moe_quantize_combine_bwd_method" -rP -s
```

## later 

```
pre-commit run pyink --files \
  src/maxtext/layers/moe.py \
  tests/unit/moe_test.py \
  src/maxtext/configs/types.py \
  src/maxtext/configs/base.yml
```