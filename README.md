MaxText
======

Attention modeling for those without it.

Install
========
manual setup:

```
bash setup.sh
```

Run
====
```
python3 train.py configs/base.yml run_name=$USER_$RANDOM
```

Status
======

- Really basic training loop on LM1B and inline inference / decoding "works".
- Multihost training works.
- Checkpointing is as yet untested.
- Absolutely nothing is yet tuned or profiled.

TODO
====

 - What decoder model variant do we actually want?
 - More flexible demo prompting / simple batch inference script.
 - Prefix-LM support for input->target datasets.
 - Should we use CLU metric helpers or hand-roll that stuff?
 - We have simple tf.data pipeline, but should we use SeqIO? Grain? an outside library?
