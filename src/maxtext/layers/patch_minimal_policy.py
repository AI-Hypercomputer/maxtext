import re

def update_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    old_names = '''    names = [
        "query_proj",
        "value_proj",
        "key_proj",
        "kv_proj",
        "qkv_proj",
        "out_proj",
        "mlpwi_0",
        "mlpwi_1",
        "mlpwi",
        "mlpwo",
    ]'''
    
    new_names = '''    names = [
        "query_proj",
        "value_proj",
        "key_proj",
        "kv_proj",
        "qkv_proj",
        "out_proj",
        "mlpwi_0",
        "mlpwi_1",
        "mlpwi",
        "mlpwo",
        "per_layer_input_gate",
        "per_layer_projection",
        "post_per_layer_input_norm",
        "per_layer_model_projection",
        "per_layer_projection_norm",
        "layer_scalar",
    ]'''
    
    if old_names in content:
        content = content.replace(old_names, new_names)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"Updated {filepath}")
    else:
        print(f"Could not find old_names in {filepath}")

update_file('/usr/local/google/home/mattdonati/common_files/maxtext/src/maxtext/layers/decoders.py')
update_file('/usr/local/google/home/mattdonati/common_files/maxtext/src/maxtext/layers/nnx_decoders.py')
