# Download the hf llama 8B Instruct model
# export CHECKPOINT_ORIGINAL=/mnt/disks/persist/checkpoints/huggingface/Llama3.1-8B-Instruct
# huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir $CHECKPOINT_ORIGINAL

# Or download the DeepSeek llama 8B model
export CHECKPOINT_ORIGINAL=/mnt/disks/persist/checkpoints/huggingface/DeepSeek-R1-Distill-Llama-8B
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-8B --local-dir $CHECKPOINT_ORIGINAL

export CHECKPOINT_TPU_SCANNED=$CHECKPOINT_ORIGINAL/scanned_chkpt
export TOKENIZER="${MAXTEXT_ASSETS_ROOT:-${MAXTEXT_PKG_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/assets/tokenizers}}"/tokenizer_llama3.tiktoken
export BASE_OUTPUT_PATH=$CHECKPOINT_ORIGINAL
export RUN_NAME=unscanned_chkpt
export CHECKPOINT_TPU_UNSCANNED=$BASE_OUTPUT_PATH/$RUN_NAME/checkpoints/0/items
export CHECKPOINT_TPU_CONVERTED_BACK=${CHECKPOINT_ORIGINAL}/converted_back
export MODEL_SIZE=llama3.1-8b
export GOLDEN_LOGITS="${MAXTEXT_TEST_ASSETS_ROOT:-${MAXTEXT_REPO_ROOT:-$PWD}/tests/assets}"/golden_logits/golden_data_deepseek_r1_distill_llama3.1_8b.jsonl

# Remove previous checkpoints to have a clean start
rm $CHECKPOINT_ORIGINAL/scanned_chkpt $CHECKPOINT_ORIGINAL/unscanned_chkpt ${CHECKPOINT_ORIGINAL}/converted_back

# Convert the checkpoints
JAX_PLATFORMS=cpu python3 -m MaxText.utils.ckpt_scripts.llama_or_mistral_ckpt --base-model-path=$CHECKPOINT_ORIGINAL --model-size=$MODEL_SIZE --maxtext-model-path=$CHECKPOINT_TPU_SCANNED  --huggingface-checkpoint=true

# Let's verify the original checkpoint to see if it matches with Huggingface golden logits
python3 -m tests.utils.forward_pass_logit_checker "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"//base.yml base_output_directory=${BASE_OUTPUT_PATH} tokenizer_path=$TOKENIZER tokenizer_type=tiktoken load_parameters_path=${CHECKPOINT_TPU_SCANNED}/0/items run_name=forward_pass_test_hf per_device_batch_size=1 model_name=$MODEL_SIZE max_prefill_predict_length=3 max_target_length=4 dataset_type=synthetic dtype=float32 activations_in_float32=true matmul_precision=float32 async_checkpointing=false scan_layers=true --max_kl_div=1e-4 --hf_model_path=$CHECKPOINT_ORIGINAL --golden_logits_path=$GOLDEN_LOGITS

# Let's verify the generated scanned checkpoint to see if it matches with Huggingface golden logits
python3 -m tests.utils.forward_pass_logit_checker "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"//base.yml base_output_directory=${BASE_OUTPUT_PATH} tokenizer_path=$TOKENIZER tokenizer_type=tiktoken load_parameters_path=${CHECKPOINT_TPU_SCANNED}/0/items run_name=forward_pass_test_hf per_device_batch_size=1 model_name=$MODEL_SIZE max_prefill_predict_length=3 max_target_length=4 dataset_type=synthetic dtype=float32 activations_in_float32=true matmul_precision=float32 async_checkpointing=false scan_layers=true --max_kl_div=1e-4 --golden_logits_path=$GOLDEN_LOGITS

# If not, we can convert the checkpoint back from MaxText to Huggingface and compare with the original one
JAX_PLATFORMS=cpu python3 -m MaxText.utils.ckpt_scripts.llama_mistral_mixtral_orbax_to_hf "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"//base.yml base_output_directory=gs://runner-maxtext-logs load_parameters_path=${CHECKPOINT_TPU_SCANNED}/0/items run_name=convert_to_hf model_name=${MODEL_SIZE} hf_model_path=$CHECKPOINT_TPU_CONVERTED_BACK

python3 -m tests.utils.hf_checkpoint_conversion_checker --original_ckpt=${CHECKPOINT_ORIGINAL} --converted_ckpt=${CHECKPOINT_TPU_CONVERTED_BACK}

# If everything looks good, we move on to convert to the unrolled checkpoint for performant serving
JAX_PLATFORMS=cpu python3 -m MaxText.generate_param_only_checkpoint "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"//base.yml async_checkpointing=false base_output_directory=${BASE_OUTPUT_PATH} load_parameters_path=${CHECKPOINT_TPU_SCANNED}/0/items run_name=${RUN_NAME} model_name=${MODEL_SIZE} force_unroll=true

# Let's verify the generated unscanned checkpoint to see if it matches with Huggingface golden logits
python3 -m tests.utils.forward_pass_logit_checker "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"//base.yml base_output_directory=${BASE_OUTPUT_PATH} tokenizer_path=$TOKENIZER tokenizer_type=tiktoken load_parameters_path=${CHECKPOINT_TPU_UNSCANNED} run_name=forward_pass_test_hf per_device_batch_size=1 model_name=$MODEL_SIZE max_prefill_predict_length=3 max_target_length=4 dataset_type=synthetic dtype=float32 activations_in_float32=true matmul_precision=float32 async_checkpointing=false scan_layers=false --max_kl_div=1e-4 --golden_logits_path=$GOLDEN_LOGITS

# Now we are good to go, serve with performance!
JAX_PLATFORMS=tpu python3 -m maxtext.decode "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"//base.yml tokenizer_path=$TOKENIZER tokenizer_type=tiktoken run_name=runner_2025-02-13-08-31 steps=10 weight_dtype=bfloat16 async_checkpointing=false model_name=$MODEL_SIZE ici_fsdp_parallelism=1 ici_autoregressive_parallelism=-1 per_device_batch_size=1 prompt="I love to" scan_layers=false load_parameters_path=$CHECKPOINT_TPU_UNSCANNED

# You can also check the results from scanned version, just double check, not necessary
JAX_PLATFORMS=tpu python3 -m maxtext.decode "${MAXTEXT_CONFIGS_DIR:-${MAXTEXT_REPO_ROOT:-$PWD}/src/maxtext/configs}"//base.yml tokenizer_path=$TOKENIZER tokenizer_type=tiktoken run_name=runner_2025-02-13-08-31 steps=10 weight_dtype=bfloat16 async_checkpointing=false model_name=$MODEL_SIZE ici_fsdp_parallelism=1 ici_autoregressive_parallelism=-1 per_device_batch_size=1 prompt="I love to" scan_layers=true load_parameters_path=$CHECKPOINT_TPU_SCANNED/0/items

##### Output from huggingface llama 8B Instruct checkpoint on MaxText:
#Input `I love to` -> ` travel and explore new places, but I also love to stay at home and relax. I'm a bit of a homebody, and I enjoy spending time with my family and friends. I'm a bit of a foodie, and I love trying new recipes and experimenting with different flavors and ingredients. I'm also a bit of a movie buff, and I love watching classic films and new releases alike.
# I'm a bit of a hopeless romantic, and I believe in the idea of true love. I'm looking for someone who shares my values and my sense of humor, and who is always up for a good time. I'm a bit of a goofball, and I love to laugh and have fun. I'm looking for someone who can keep up with me and appreciate my quirks.
# I'm a bit of a creative person, and I love to express myself through art, music, and writing. I'm a bit of a dreamer, and I love to imagine new possibilities and scenarios. I'm a bit of a perfectionist, and I always strive to do my best and be my best self.
# ...

##### Output from huggingface DeekSeek distilled llama 8B checkpoint on MaxText:
# Input `I love to` -> ` write, but I'm not sure how to start a blog. I have some ideas, but I don't know where to begin. Maybe I should just start writing and see where it goes. But I don't want to end up with just a few posts and then stop. How do I keep going?

# I also wonder if I should focus on a specific niche or write about a variety of topics. I like cooking, photography, and personal development. It's hard to decide which one to focus on. Maybe I can combine them somehow?

# Another thing is, I'm not sure about the technical stuff. Do I need to buy a domain name? What about hosting? I've heard about platforms like WordPress, but I don't know how to set it up. Is it hard to learn?

# And then there's the whole SEO thing. I know it's important for getting traffic, but I don't understand how to optimize my blog for search engines. Maybe I can figure it out as I go, but I don't want to spend too much time on it.

# Monetization is another concern. I don't want ads taking over my blog, but I also don't want to miss out on earning potential. Affiliate marketing sounds interesting, but I don't know how to start with that either.

# I also need to think about branding. How do I create a unique voice and style for my blog? I don't want it to look like everyone else's. Maybe I can experiment with different styles and see what works.

# Oh, and engagement. I want my readers to stick around and come back, but I'm not sure how to encourage that. Should I comment on other blogs? Share my posts on social media? Maybe join some online communities?

# I guess I need a plan. Maybe I should start with a specific niche to build a solid foundation. But how do I choose which one? I don't want to limit myself too early, but I also don't want to spread myself too thin.

# I should also set some goals. How often should I post? What kind of content should I create? Maybe I can start with a weekly post and see how that goes. And as I learn more about SEO and monetization, I can incorporate those strategies.

# Wait, but what if I don't have time to keep up with a regular schedule? I need to make sure I can commit to this in the long run. Maybe I can set a realistic schedule that I can stick to without getting overwhelmed.

# Another thought: I should probably start with a simple setup. Use a free blogging platform to get the hang of it before investing in a domain name and hosting. That way, I can test my ideas and see what works without any pressure.

# I also need to think about the legal aspects. Do I need to register my blog or get any licenses? I don't want to run into any trouble down the line, so maybe I should look into that.

# Overall, I think the key is to start small, focus on one area, and gradually build from there. I can always expand my niche or add more topics as I grow. And I should be patient, knowing that it might take time to see results. I just need to stay motivated and keep learning as I go.
# <|reserved_special_token_9|>

# Starting a blog can be an exciting journey, but it's important to approach it thoughtfully. Here's a structured plan based on your considerations:

# ### 1. **Choose a Niche**
#    - **Start Small:** Begin with a specific niche you're passionate about. Consider your interests: cooking, photography, or personal development. You can always expand later.
#    - **Combine Interests:** If you're unsure, try blending your passions. For example, a food blog with photography tips or a personal development blog with recipes.

# ### 2. **Set a Publishing Schedule**
#    - **Start Regularly:** Commit to posting weekly. This helps build consistency without overwhelming you.
#    - **Adjust as Needed:** If a schedule becomes too demanding, be flexible and adjust it to fit your lifestyle.

# ### 3. **Technical Setup**
#    - **Use Free Platforms:** Begin with a free blogging platform like WordPress.com or Medium to get familiar with the basics without initial costs.
#    - **Gradual Transition:** Once you're comfortable, consider moving to self-hosted WordPress with a domain name and hosting. Research affordable options like Bluehost or SiteGround.

# ### 4. **Learn SEO Gradually**
#    - **Start Simple:** Focus on basic SEO practices, such as using relevant keywords and internal linking. Tools like Google Search Console can help.
#    - **Invest Time:** Dedicate time to learn SEO strategies, perhaps through online courses or guides, as your blog grows.

# ### 5. **Monetization Strategies**
#    - **Affiliate Marketing:** Start with affiliate programs that align with your niche. Sign up for programs like Amazon Associates or ShareASale.
#    - **Ad Revenue:** Once your blog gains traction, explore ad networks like Google AdSense. Ensure ads don't disrupt your content.

# ### 6. **Branding and Voice**
#    - **Experiment:** Find your unique voice and style by experimenting with different formats and tones. Engage with your audience to see what resonates.
#    - **Visual Identity:** Develop a consistent color scheme, fonts, and logo as your brand evolves.

# ### 7. **Engagement Strategies**
#    - **Social Media:** Share your posts on platforms like Instagram, Pinterest, or Twitter to reach a broader audience.
#    - **Community Building:** Engage with other bloggers by commenting on their posts and joining online communities (e.g., Reddit forums or Facebook groups).

# ### 8. **Legal Considerations**
#    - **Copyright and Licensing:** Protect your content with proper copyright notices and licenses. Consider using Creative Commons for shared content.

# ### 9. **Stay Motivated and Learn**
#    - **Set Realistic Goals:** Celebrate small milestones to stay motivated.
#    - **Continuous Learning:** Keep up with trends and improve your skills over time.

# ### 10. **Long-Term Commitment**
#    - **Build a Community:** Focus on fostering relationships with your readers to create a loyal audience.
#    - **Adapt and Grow:** Be open to changing your approach as you learn and grow with your blog.

# By following this plan, you can build a meaningful blog that evolves over time, reflecting your growth and interests. Remember, it's a journey, so enjoy the process and stay adaptable.<|end_of_text|><|begin_of_text|>://

# ---

# **Final Answer:**

# Starting a blog is an exciting venture that allows you to share your passions and ideas with the world. Here's a structured approach to help you get started:

# ### 1. **Choose a Niche**
#    - **Start Small:** Begin with a specific niche that aligns with your interests, such as cooking, photography, or personal development.
#    - **Combine Interests:** If you're unsure, consider blending your passions, like a food blog with photography tips or a personal development blog with recipes.

# ### 2. **Set a Publishing Schedule**
#    - **Start Regularly:** Commit to posting once a week to build consistency without overwhelming yourself.
#    - **Adjust as Needed:** Be flexible and adjust your schedule if it becomes too demanding.

# ### 3. **Technical Setup**
#    - **Use Free Platforms:** Begin with free platforms like WordPress.com or Medium to get familiar with the basics.
#    - **Transition Gradually:** Once comfortable, consider moving to self-hosted WordPress with a domain name and hosting.

# ### 4. **Learn SEO Gradually**
#    - **Start Simple:** Focus on basic SEO practices, such as using relevant keywords and internal linking.
#    - **Invest Time:** Dedicate time to learn SEO strategies through online courses or guides.

# ### 5. **Monetization Strategies**
#    - **Affiliate Marketing:** Sign up for affiliate programs like Amazon Associates or ShareASale.
#    - **Ad Revenue:** Explore ad networks like Google AdSense once your blog gains traction.

# ### 6. **Branding and Voice**
#    - **Experiment:** Find your unique voice and style by experimenting with different formats and tones.
#    - **Visual Identity:** Develop a consistent brand with a color scheme, fonts, and logo.

# ### 7. **Engagement Strategies**
#    - **Social Media:** Share your posts on platforms like Instagram, Pinterest, or Twitter.
#    - **Community Building:** Engage with other bloggers and join online communities.

# ### 8. **Legal Considerations**
#    - **Copyright and Licensing:** Protect your content with proper notices and licenses.

# ### 9. **Stay Motivated and Learn**
#    - **Set Realistic Goals:** Celebrate small milestones to stay motivated.
#    - **Continuous Learning:** Keep up with trends and improve your skills.

# ### 10. **Long-Term Commitment**
#    - **Build a Community:** Focus on fostering relationships with your readers.
#    - **Adapt and Grow:** Be open to changing your approach as you grow.

# By following this plan, you can build a meaningful blog that evolves over time, reflecting your growth and interests. Enjoy the journey and stay adaptable as you explore this exciting world of blogging!<|end_of_text|><|begin_of_text|>://

# ---

# **Final Answer:**

# Starting a blog is an exciting journey that allows you to share your passions and ideas with the world. Here's a structured approach to help you get started:

# ### 1. **Choose a Niche**
#    - **Start Small:** Begin with a specific niche that aligns with your interests, such as cooking, photography, or personal development.
#    - **Combine Interests:** If you're unsure, consider blending your passions, like a food blog with photography tips or a personal development blog with`

##### Output from the original Huggingface model 8B on GPU:
# [{'generated_text': "I love to cook, but my problem is that I don't have enough space in my kitchen. I'm trying to figure out how to make the most of the limited space I have. I have a small kitchen, so I need to be strategic about how I use the existing space. Let me think about the different areas in my kitchen and how I can optimize them.\n\nFirst, my countertops are pretty cluttered. I have appliances like my coffee maker, toaster, and blender all out in the open. I wonder if I can find a better way to store these so they don't take up as much space. Maybe I can get some sort of organizers or racks to keep them in one place.\n\nThen, my cabinets are mostly empty, but I don't have enough storage for all my spices and utensils. I have a lot of spices in those small glass containers, and they take up a lot of space. Maybe I can get some spice racks or use some of those small containers to organize them better. Also, my utensils are all jumbled up in a drawer; I should probably get some dividers or a drawer organizer to keep them separate.\n\nI have a small pantry, but it's not very organized. I have canned goods and dry goods all mixed together. I think I need some kind of shelf or bin to separate them into categories like baking supplies, snacks, and pantry staples. That way, I can find what I need without digging through everything.\n\nThe sink area is another issue. I always have dishes piling up next to the sink. I should get some dish racks or maybe a small trash can with a lid to keep things tidy. Also, there's a lot of water spots on the counter; maybe a wet wipe container would help keep things clean.\n\nI have a small dining area that's also cluttered. I have a small table and chairs, but there's always stuff laying on the table. I should probably get some storage benches or a shelf to put things like mail or coats out of the way.\n\nIn the corner where the stove is, there's not much space. I have a pan rack there, but it's not enough. I might need a larger rack or maybe some hanging storage solutions to maximize the vertical space.\n\nThe floor is another area I can utilize. I have some extra boxes under the sink that I can maybe move elsewhere or get some stacking bins to store things like seasonal items or extra kitchen gadgets.\n\nI also have a lot of small kitchen appliances that I don't use often. Maybe I can find a place to store them more efficiently, like on the top shelf of the pantry or using some vertical storage.\n\nI think I need to focus on each area one by one and figure out the best way to organize them. Maybe start with the countertops by getting some organizers for the appliances. Then move on to the cabinets and pantry, organizing spices and utensils. After that, work on the sink area and dining area. Finally, look into the vertical spaces and floor storage to maximize the space.\n\nI should also think about multi-functional furniture, like a bench that can hold some storage or a kitchen cart that can help move things around without taking up too much space. Maybe using the walls for more storage with some hooks or shelves.\n\nIt's a bit overwhelming, but if I take it step by step, I can definitely make a big difference in my kitchen. I'll start by decluttering and then figure out the best storage solutions for each area.\n**Step-by-Step Explanation and Answer:**\n\nTo optimize your small kitchen, follow this organized approach:\n\n1. **Countertops:**\n   - **Clear Clutter:** Remove appliances like the coffee maker, toaster, and blender.\n   - **Storage Solutions:**\n     - Use a wall-mounted organizer or a shelf to keep these appliances tidy.\n     - Opt for a slim drawer or tray for small items like cutlery or utensils.\n\n2. **Cabinets:**\n   - **Spices:** Install a spice rack or use small, labeled containers for easy access.\n   - **Utensils:** Utilize drawer dividers or a kitchen utensil organizer to keep them sorted.\n\n3. **Pantry:**\n   - **Categorize Foods:** Use labeled bins or shelves for baking supplies, snacks, and staples.\n   - **Vertical Storage:** Install shelving or use hanging storage for frequently used items.\n\n4. **Sink Area:**\n   - **Dish Storage:** Place a dish rack near the sink for drying utensils or dishes.\n   - **Trash and Cleaning Supplies:** Keep a small trash can with a lid and store cleaning products on a shelf or under the sink.\n\n5. **Dining Area:**\n   - **Storage Solutions:**\n     - Use a small bench with storage for items like mail or coats.\n     - Install a shelf above the table for decor or extra storage.\n\n6. **"}]
