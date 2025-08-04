<!--
 # SPDX-License-Identifier: Apache-2.0
 -->

# High Performance Model Configs on TPU v5p
Expected performance results for 32B, 64B, 128B, 256B, 512B, and 1024B parameter models running on TPU v5p:

### 32B model
| Hardware    | TFLOP/sec/chip     | MFU     | 
| ----------- | -----------------: | ------- | 
| 1x v5p-128  | 328                | 71.47%  |
| 2x v5p-128  | 319                | 69.43%  |

### 64B model
| Hardware    | TFLOP/sec/chip     | MFU     | 
| ----------- | -----------------: | ------- | 
| 1x v5p-128  | 323                | 70.31%  |
| 2x v5p-128  | 309                | 67.26%  |

### 128B model
| Hardware    | TFLOP/sec/chip     | MFU     | 
| ----------- | -----------------: | ------- | 
| 1x v5p-256  | 315                | 68.68%  |
| 2x v5p-256  | 305                | 66.34%  |
| 1x v5p-512  | 316                | 68.83%  |
| 2x v5p-512  | 274                | 59.66%  |

### 256B model
| Hardware    | TFLOP/sec/chip     | MFU     | 
| ----------- | -----------------: | ------- | 
| 1x v5p-1024 | 308                | 67.09%  |
| 2x v5p-1024 | 289                | 62.99%  |

### 512B model
| Hardware    | TFLOP/sec/chip     | MFU     | 
| ----------- | -----------------: | ------- | 
| 1x v5p-1024 | 294                | 63.99%  |
| 2x v5p-1024 | 282                | 61.45%  |

### 1024B model
| Hardware    | TFLOP/sec/chip     | MFU     | 
| ----------- | -----------------: | ------- | 
| 1x v5p-2048 | 254                | 55.35%  |
| 2x v5p-2048 | 237                | 51.55%  |
| 1x v5p-4096 | 297                | 64.80%  |
| 2x v5p-4096 | 279                | 60.83%  |