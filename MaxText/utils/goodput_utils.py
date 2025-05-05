"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import contextlib

import jax

from MaxText import max_logging
from ml_goodput_measurement import goodput, monitoring


def maybe_start_goodput_monitoring(config):
    if config.monitor_goodput and jax.process_index() == 0:
        goodput_monitor = monitoring.GoodputMonitor(
            job_name=config.run_name,
            logger_name=f"goodput_{config.run_name}",
            tensorboard_dir=config.tensorboard_dir,
            upload_interval=config.goodput_upload_interval_seconds,
            monitoring_enabled=True,
            pathway_enabled=config.enable_pathways_goodput,
            include_badput_breakdown=True,
            include_step_deviation=config.monitor_step_time_deviation,
            step_deviation_interval_seconds=config.step_deviation_interval_seconds,
        )
        goodput_monitor.start_goodput_uploader()
        max_logging.log("Started Goodput upload to Tensorboard in the background")

        if config.monitor_step_time_deviation:
            goodput_monitor.start_step_deviation_uploader()
            max_logging.log("Started step time deviation upload to Tensorboard in the background")


def create_goodput_recorder(config):
    logger_name = f"goodput_{config.run_name}"
    enable_logging = config.enable_goodput_recording and jax.process_index() == 0
    recorder = goodput.GoodputRecorder(config.run_name, logger_name, enable_logging)
    return recorder


@contextlib.contextmanager
def maybe_record_goodput(recorder, base_event_name: str, *start_args, **start_kwargs):
    """
    Functional context manager to call start/end methods on a recorder.

    Directly calls methods like `recorder.event_start_time(*args)` and
    `recorder.event_end_time()`. Handles None recorders and checks the
    `enable_goodput_recording` flag in the config.

    Args:
        recorder: The recorder instance with methods like '{base_event_name}_start_time',
                  '{base_event_name}_end_time', or None.
        config: A configuration object/dict expected to have an
                `enable_goodput_recording` attribute/key (boolean).
        base_event_name (str): The base name for the event (e.g., "record_job").
        *start_args: Optional positional arguments to pass ONLY to the start method.
        **start_kwargs: Optional keyword arguments to pass ONLY to the start method.
                       (End method is assumed to take no args for simplicity,
                        but could be extended if needed).

    Yields:
        The recorder instance (or None) for potential use within the 'with' block.
    """
    start_method = None
    end_method = None

    if recorder:
        start_method_name = f"record_{base_event_name}_start_time"
        end_method_name = f"record_{base_event_name}_end_time"

        start_method = getattr(recorder, start_method_name, None)
        end_method = getattr(recorder, end_method_name, None)
        assert start_method

        start_method(*start_args, **start_kwargs)

    try:
        # yield the recorder (or None if disabled)
        yield recorder
    finally:
        # there are cases where there exists no end method but we'll use this context manager
        # regardless, for consistency and simplicity
        if recorder and end_method:
            end_method()
