export CHECKPOINT_ORIGINAL=/mnt/disks/persist/checkpoints/huggingface/Llama3.1-70B-Instruct
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-70B --local-dir $CHECKPOINT_ORIGINAL

export CHECKPOINT_TPU_SCANNED=$CHECKPOINT_ORIGINAL/scanned_chkpt

export TOKENIZER="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_REPO_ROOT:-$PWD}/assets}"/tokenizer_llama3.tiktoken
export BASE_OUTPUT_PATH=$CHECKPOINT_ORIGINAL
export RUN_NAME=unscanned_chkpt
export CHECKPOINT_TPU_UNSCANNED=$BASE_OUTPUT_PATH/$RUN_NAME/checkpoints/0/items
export MODEL_SIZE=llama3.1-70b
export GOLDEN_LOGITS="${MAXTEXT_TEST_ASSETS_ROOT:-${MAXTEXT_REPO_ROOT:-$PWD}/test_assets}"/golden_data_deepseek_r1_distill_llama3.1_70b.jsonl

# Remove previous checkpoints to have a clean start
rm $CHECKPOINT_ORIGINAL/scanned_chkpt $CHECKPOINT_ORIGINAL/unscanned_chkpt ${CHECKPOINT_ORIGINAL}/converted_back

JAX_PLATFORMS=cpu python3 -m MaxText.llama_or_mistral_ckpt --base-model-path=$CHECKPOINT_ORIGINAL --model-size=$MODEL_SIZE --src/MaxText-model-path=$CHECKPOINT_TPU_SCANNED  --huggingface-checkpoint=true

# Let's verify the original checkpoint to see if it matches with Huggingface golden logits
python3 -m tests.forward_pass_logit_checker "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} tokenizer_path=$TOKENIZER tokenizer_type=tiktoken load_parameters_path=${CHECKPOINT_TPU_SCANNED}/0/items run_name=forward_pass_test_hf per_device_batch_size=1 model_name=$MODEL_SIZE max_prefill_predict_length=3 max_target_length=4 dataset_type=synthetic dtype=float32 activations_in_float32=true matmul_precision=float32 async_checkpointing=false scan_layers=true --max_kl_div=1e-4 --hf_model_path=$CHECKPOINT_ORIGINAL --golden_logits_path=$GOLDEN_LOGITS

# Let's verify the generated scanned checkpoint to see if it matches with Huggingface golden logits
python3 -m tests.forward_pass_logit_checker "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} tokenizer_path=$TOKENIZER tokenizer_type=tiktoken load_parameters_path=${CHECKPOINT_TPU_SCANNED}/0/items run_name=forward_pass_test_hf per_device_batch_size=1 model_name=$MODEL_SIZE max_prefill_predict_length=3 max_target_length=4 dataset_type=synthetic dtype=float32 activations_in_float32=true matmul_precision=float32 async_checkpointing=false scan_layers=true --max_kl_div=1e-4 --golden_logits_path=$GOLDEN_LOGITS

# If not, we can convert the checkpoint back from MaxText to Huggingface and compare with the original one
JAX_PLATFORMS=cpu python3 -m MaxText.llama_mistral_mixtral_orbax_to_hf "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/base.yml base_output_directory=gs://runner-src/MaxText-logs load_parameters_path=${CHECKPOINT_TPU_SCANNED}/0/items run_name=convert_to_hf model_name=${MODEL_SIZE} hf_model_path=$CHECKPOINT_TPU_CONVERTED_BACK

python3 -m tests.hf_checkpoint_conversion_checker --original_ckpt=${CHECKPOINT_ORIGINAL} --converted_ckpt=${CHECKPOINT_TPU_CONVERTED_BACK}

# If everything looks good, we move on to convert to the unrolled checkpoint for performant serving
JAX_PLATFORMS=cpu python3 -m MaxText.generate_param_only_checkpoint "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/base.yml async_checkpointing=false base_output_directory=${BASE_OUTPUT_PATH} load_parameters_path=${CHECKPOINT_TPU_SCANNED}/0/items run_name=${RUN_NAME} model_name=${MODEL_SIZE} force_unroll=true

# Let's verify the generated unscanned checkpoint to see if it matches with Huggingface golden logits
python3 -m tests.forward_pass_logit_checker "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} tokenizer_path=$TOKENIZER tokenizer_type=tiktoken load_parameters_path=${CHECKPOINT_TPU_UNSCANNED} run_name=forward_pass_test_hf per_device_batch_size=1 model_name=$MODEL_SIZE max_prefill_predict_length=3 max_target_length=4 dataset_type=synthetic dtype=float32 activations_in_float32=true matmul_precision=float32 async_checkpointing=false scan_layers=false --max_kl_div=1e-4 --golden_logits_path=$GOLDEN_LOGITS

# Now we are good to go, serve with performance!
JAX_PLATFORMS=tpu python3 -m MaxText.decode "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/base.yml tokenizer_path=$TOKENIZER tokenizer_type=tiktoken run_name=runner_2025-02-13-08-31 steps=10 weight_dtype=bfloat16 async_checkpointing=false model_name=$MODEL_SIZE ici_fsdp_parallelism=1 ici_autoregressive_parallelism=-1 per_device_batch_size=1 prompt="I love to" scan_layers=false load_parameters_path=$CHECKPOINT_TPU_UNSCANNED

# You can also check the results from scanned version, just double check, not necessary
JAX_PLATFORMS=tpu python3 -m MaxText.decode "${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/MaxText}/"configs/base.yml tokenizer_path=$TOKENIZER tokenizer_type=tiktoken run_name=runner_2025-02-13-08-31 steps=10 weight_dtype=bfloat16 async_checkpointing=false model_name=$MODEL_SIZE ici_fsdp_parallelism=1 ici_autoregressive_parallelism=-1 per_device_batch_size=1 prompt="I love to" scan_layers=true load_parameters_path=$CHECKPOINT_TPU_SCANNED/0/items

# Example output
# Input `I love to` -> ` read, but I don't have much time. How can I read more books?
# I'm a busy person, but I want to read more. How can I fit reading into my schedule?
# I want to read more, but I don't have enough time. What can I do?
# I don't have time to read, but I want to. How can I make time for reading?
# I want to read more, but my schedule is too busy. What can I do?
# I don't have much time, but I want to read more. How can I manage that?
# I want to read more, but I'm too busy. How can I fit reading into my schedule?
# I don't have time to read, but I want to. What can I do?
# I want to read more, but I don't have enough time. How can I make time for reading?
# I'm busy, but I want to read more. How can I fit reading into my schedule?

# Okay, so I'm trying to figure out how to read more books even though I'm really busy. I love reading, but it seems like I never have the time. Let me think about this step by step.

# First, maybe I can start by looking at my daily routine to see where I can squeeze in some reading time. I usually wake up early, get ready for work, and then have a busy day. Maybe I can wake up a bit earlier each day to read before work. But wait, I'm not a morning person. That might be tough. Maybe I can try just 15 minutes in the morning. That doesn't seem too bad.

# Another idea is to use my commute. I take the bus to work, which is about 30 minutes each way. I could listen to audiobooks or read on my phone during that time. I've heard that audiobooks are a good way to consume books quickly, especially for non-fiction. But I'm more into fiction, so maybe I can find a good fiction audiobook. Or maybe I can read e-books on my phone. I have a Kindle app, so that could work.

# Lunch breaks are another possibility. I usually have about an hour for lunch. Maybe I can spend 20-30 minutes reading during that time. But sometimes I meet friends or have meetings, so it might not be consistent. Still, it's worth trying on the days I'm alone.

# Evenings are tricky because I'm often tired after work. But maybe right before bed, I can read for 15-20 minutes instead of scrolling through my phone. That might also help me wind down and sleep better. Plus, it's a good way to relax.

# I also have weekends. Maybe I can dedicate a couple of hours on Saturday or Sunday to reading. That could help me catch up on my reading without feeling rushed.

# Another thought: maybe I can make a reading list and set a goal for how many books I want to read each month. That way, I can track my progress and stay motivated. I could use a reading log or an app to keep track.

# I should also consider the types of books I'm reading. Maybe shorter books or novellas can be finished quicker, fitting into my busy schedule better. Or I could mix in some graphic novels, which are usually faster to read.

# I've heard about the concept of "reading sprints" where you read for a set amount of time without distractions. Maybe I can try that during my breaks or in the evenings. It might help me focus and get through more pages.

# Another idea is to join a book club or find a reading buddy. That could keep me accountable and give me a reason to prioritize reading. Plus, discussing books with others might make it more enjoyable and motivate me to keep going.

# I also need to think about eliminating distractions. Maybe turning off notifications on my phone or finding a quiet spot where I can read without interruptions. Creating a dedicated reading space might help me get into the right mindset.

# What about multitasking? I could listen to audiobooks while doing chores, exercising, or driving. That way, I'm making use of time that would otherwise be unproductive.

# I should also be realistic about my goals. I might not be able to read as much as I'd like, but setting achievable targets can help me stay on track without feeling overwhelmed. Maybe start with one book a month and gradually increase as I find more time.

# Another thing to consider is the format of the books. E-books are convenient because I can carry them on my phone, but physical books might be better for certain times, like before bed when I want to avoid screens.

# I could also try speed reading techniques, but I'm not sure how effective they are. Maybe skimming through less important parts or focusing on key points could help me get through books faster.

# Lastly, I need to prioritize reading as a form of self-care. It's important for my mental health and relaxation, so making time for it should be non-negotiable, just like other important activities.

# Putting it all together, I think the key is to find small pockets of time throughout the day and use them effectively. Whether it's during commutes, breaks, or before bed, every little bit counts. Combining different strategies like audiobooks, e-books, setting goals, and creating a reading-friendly environment can help me read more despite being busy.
# <|reserved_special_token_9|>

# To read more despite a busy schedule, consider the following organized approach:

# 1. **Morning Routine**: Start with 15 minutes of reading in the morning, even if you're not a morning person. It sets a positive tone for the day.

# 2. **Commute Utilization**: Use your 30-minute bus commute to listen to audiobooks or read e-books on your phone. This is an efficient way to consume books, especially fiction.

# 3. **Lunch Breaks**: Dedicate 20-30 minutes of your lunch break to reading, especially on days when you're alone. This provides a midday mental break.

# 4. **Evening Routine**: Wind down before bed with 15-20 minutes of reading instead of screen time. This aids relaxation and sleep.

# 5. **Weekend Dedication**: Allocate a couple of hours on weekends to reading, allowing you to catch up without feeling rushed.

# 6. **Reading Goals and Tracking**: Create a reading list and set monthly goals. Use a reading log or app to track progress and stay motivated.

# 7. **Book Selection**: Opt for shorter books, novellas, or graphic novels to fit into your schedule and vary your reading material.

# 8. **Reading Sprints**: Try focused reading sessions without distractions during breaks or evenings to maximize productivity.

# 9. **Accountability and Community**: Join a book club or find a reading buddy for accountability and enjoyment. Discussions can enhance your reading experience.

# 10. **Distraction Management**: Create a quiet reading space and minimize interruptions by turning off notifications.

# 11. **Multitasking with Audiobooks**: Listen to audiobooks during chores, exercise, or driving to utilize otherwise idle time.

# 12. **Realistic Goal Setting**: Start with achievable targets, like one book a month, and gradually increase as you find more time.

# 13. **Book Format Flexibility**: Use e-books for convenience and physical books for screen-free reading, especially before bed.

# 14. **Speed Reading Techniques**: Experiment with skimming or focusing on key points to read more efficiently.

# 15. **Prioritize Self-Care**: Treat reading as essential for mental health and relaxation, making it a non-negotiable part of your routine.

# By integrating these strategies, you can effectively use small time pockets to read more, combining audiobooks, e-books, goal setting, and a conducive reading environment.<|end_of_text|>
