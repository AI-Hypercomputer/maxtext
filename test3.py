from MaxText import checkpointing
checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
    #"tmp/test",
    "gs://shuningjin-multipod-dev/test",
    enable_checkpointing=True,
    use_async=False,  # Synchronous saving for simplicity in conversion script
    save_interval_steps=1,  # Save at step 0
    use_ocdbt=True,
    use_zarr3=True,
)