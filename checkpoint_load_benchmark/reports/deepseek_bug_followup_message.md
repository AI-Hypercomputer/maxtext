Hi Abhinav, thanks for the heads-up! We checked our tested branch and confirmed our experiments were run on commit `5b6773c9b` (`maxtext-v0.2.6`, right after PR #5828 / cl/1017682327 and before those 3 recent `HF streaming` CLs), so our runs weren't affected by that regression.

Here is the Buganizer issue with the exact `pyconfig` flags and topologies we tried across the large DeepSeek models (`deepseek4-284b`, `deepseek3-671b`, `deepseek3.2-671b`, `deepseek4-1600b`) for supporting on-the-fly `FP8` / `MXFP4` → `BF16` dequantization in `safetensors_dynamic`: http://b/586746548
