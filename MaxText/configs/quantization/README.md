# Mixed precision quantization configs (currently supported for inference only).

This directory contains sample json files representing mixed precision quantization configs.

A mixed precision config json is contains the following:
    Keys represent a regex for the layer to which the config is applied.
    Values represent the quantization config for the corresponding layer.

The quantization config for any layer is defined using the following variables.
    w_bits: Number of bits used for weights, default None (implying no quantization)
    a_bits: Number of bits used for activations, default None (implying no quantization)
    w_scale: Clipping scale for weights, default 1.0
    a_scale: Clipping scale for activations, default 1.0
    tile_size: tile size for subchannel, default -1 (implying no subchannel)

For example, the config below implies 4-bit weight_only quantization for layers wi_0 and w0.
    {
    ".*/wi_0": {"w_bits": 4},
    ".*/wo": {"w_bits": 4}
    }

A special key '__default__'  can be used to override the default values.
For example the following config defines 8-bit weight only quantization for all layers.
{
  "__default__": {"w_bits": 8}
}

# To configure mixed precision quantization, define the following.
    1. A json file (e.g. example.json) in this directory with desired config
    2. Set the following parameters defined in base.yml
        quantization="intmp"
        quant_cfg_path="<path_to_config_dir>/example.json"
