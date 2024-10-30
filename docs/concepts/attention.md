# Attention

MaxText supports multiple optimization options for attention calculations. The default is [flash](https://github.com/Dao-AILab/flash-attention). This has been optimized for maximum performance.

MaxText supports the following values for the `attention` parameter:

- `flash`: Default and most performant. This is written in Pallas to achieve maximum performance.
- `dot_product`: Works with older versions of TPU, e.g v2,v3. Should be used when flash does not work.
- `cudnn_flash_te`: This is a GPU specific setting.
