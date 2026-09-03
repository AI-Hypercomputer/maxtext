"""Internal wrapper for train_distill."""

import sys
from absl import app

from maxtext.input_pipeline import grain_data_processing_internal
sys.modules["maxtext.src.maxtext.input_pipeline.grain_data_processing"] = grain_data_processing_internal

from maxtext.trainers.post_train.distillation import train_distill  # type: ignore
# Placeholder: internal



# Patch MaxTextDistillationTrainer._train_step to align with internal Tunix PeftTrainer signature.
_original_train_step = train_distill.MaxTextDistillationTrainer._train_step

def _patched_train_step(self, model, optimizer, grad_accumulator, inputs, is_update_step, **kwargs):
    # Route 'inputs' to 'inputs' and ignore grad_accumulator since MaxText doesn't support it natively.
    return _original_train_step(
        self,
        model=model,
        optimizer=optimizer,
        inputs=inputs,
        grad_accumulator=grad_accumulator,
        is_update_step=is_update_step,
        **kwargs
    )

train_distill.MaxTextDistillationTrainer._train_step = _patched_train_step  # pyrefly: ignore[bad-assignment]

if __name__ == "__main__":
    maxtext_google.g3_multiprocessing_handle_main(train_distill.main)
    app.run(train_distill.main)
