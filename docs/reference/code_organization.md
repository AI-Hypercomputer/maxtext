# MaxText Code Organization

Maxtext is purely written in JAX and python. Below are some folders and files
that show a high-level organization of the code and some key files.

File/Folder | Description
---------|---------------------------------
 `configs` | Folder contains all the config file, including model configs (llama2, mistral etc) , and pre-optimized configs for different model size on different TPUs 
 `input_pipelines` | Input training data related code 
 `layers` | Model layer implementation 
 `end_to_end` | Example scripts to run Maxtext 
 `Maxtext/train.py` | The main training script you will run directly 
 `Maxtext/config/base.yaml` | The base configuration file containing all the related info: checkpointing, model arch, sharding schema, data input, learning rate, profile, compilation, decode 
 `Maxtext/decode.py` | This is a script to run offline inference with a sample prompt
 `setup.sh`| Bash script used to install all needed library dependencies. 
