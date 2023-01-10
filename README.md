MaxText
======

An LLM codebase that is:
* high-performance
* infinitely scalable
* open-source
* simple and easily forkable
* well-tested
for Cloud TPU customers

Install
========
manual setup:

```
bash setup.sh
```

Run
====
To run a training job:
```
python3 MaxText/train.py MaxText/configs/base.yml run_name=$USER_$(date "+%Y-%m-%d-%H:%M:%S")
```

To lint:
```
pylint MaxText/
```

Tensorboard: run Tensorboard on the host TPUVM via VSCode. You will need to run the following command
```
#https://stackoverflow.com/questions/40830085/tensorboard-can-not-read-summaries-on-google-cloud-storage
gcloud auth application-default login
```


Status/TODO
======
[Backlog](http://go/maxtext-backlog)
