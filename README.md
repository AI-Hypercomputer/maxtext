MaxText
======

Attention modeling for those without it.

Install
========
manual setup:

```
sudo apt install python3.9-venv
python3.9 -m venv py39
source py39/bin/activate
pip install -U pip wheel
pip install jupyter jupyterlab ipython matplotlib clu tensorflow-text sentencepiece
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# need recent flax
git clone --branch=main https://github.com/google/flax.git
pip install -e flax
git clone --branch=main https://github.com/rwitten/MaxText.git
pip install -e adhd
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
