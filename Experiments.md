1. 872M=.87B model.
* Trained .87 billion parameter model on 150k batches (with 4 devices, 32 batches per device, sequence length of 1024) or about 19.7B examples.
* Loss of ~2.5, aligned well with Chinchilla 1B.
* Code: https://github.com/rwitten/MaxText/tree/rwitten_20230209_scale1_training
* Run: tensorboard --logdir=gs://max-experiments/20230209_scale1_scanlayersfalse_warmup/tensorboard/
