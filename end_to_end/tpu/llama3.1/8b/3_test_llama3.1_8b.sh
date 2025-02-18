# Download the hf llama 8B Instruct model
# export CHECKPOINT_ORIGINAL=/mnt/disks/persist/checkpoints/huggingface/Llama3.1-8B-Instruct
# huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir $CHECKPOINT_ORIGINAL

# Or download the DeepSeek llama 8B model
export CHECKPOINT_ORIGINAL=/mnt/disks/persist/checkpoints/huggingface/Llama3.1-8B-Instruct
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-8B --local-dir $CHECKPOINT_ORIGINAL

export CHECKPOINT_TPU_SCANNED=$CHECKPOINT_ORIGINAL/scanned_chkpt

export TOKENIZER=assets/tokenizer_llama3.tiktoken
export BASE_OUTPUT_PATH=$CHECKPOINT_ORIGINAL
export RUN_NAME=unscanned_chkpt
export CHECKPOINT_TPU_UNSCANNED=$BASE_OUTPUT_PATH/$RUN_NAME/checkpoints/0/items

JAX_PLATFORMS=cpu python3 MaxText/llama_or_mistral_ckpt.py --base-model-path=$CHECKPOINT_ORIGINAL --model-size=llama3.1-8b --maxtext-model-path=$CHECKPOINT_TPU_SCANNED  --huggingface-checkpoint=true

# Let's verify the generated scanned checkpoint to see if it matches with Huggingface golden logits
python3 MaxText/tests/forward_pass_logit_checker.py MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} tokenizer_path=$TOKENIZER load_parameters_path=${CHECKPOINT_TPU_SCANNED}/0/items run_name=forward_pass_test_hf per_device_batch_size=1 model_name=$MODEL_SIZE max_prefill_predict_length=3 max_target_length=4 dataset_type=synthetic dtype=float32 activations_in_float32=true matmul_precision=float32 async_checkpointing=false scan_layers=true --max_kl_div=1e-4

# If not, we can convert the checkpoint back from MaxText to Huggingface and compare with the orignal one
JAX_PLATFORMS=cpu python3 MaxText/llama_mistral_mixtral_orbax_to_hf.py MaxText/configs/base.yml base_output_directory=gs://runner-maxtext-logs load_parameters_path=${CHECKPOINT_TPU_SCANNED}/0/items run_name=convert_to_hf model_name=${MODEL_SIZE} hf_model_path=/tmp/hf_llama3_1
python3 MaxText load_checkpoint.py

# If everything looks good, we move on to convert to the unrolled checkpoint for performant serving
JAX_PLATFORMS=cpu python MaxText/generate_param_only_checkpoint.py MaxText/configs/base.yml async_checkpointing=false base_output_directory=${BASE_OUTPUT_PATH} load_parameters_path=${CHECKPOINT_TPU_SCANNED}/0/items run_name=${RUN_NAME} model_name=${MODEL_SIZE} force_unroll=true


JAX_PLATFORMS=tpu python MaxText/decode.py MaxText/configs/base.yml tokenizer_path=$TOKENIZER run_name=runner_2025-02-13-08-31 steps=10 weight_dtype=bfloat16 async_checkpointing=false model_name=$MODEL_SIZE ici_fsdp_parallelism=1 ici_autoregressive_parallelism=-1 per_device_batch_size=1 prompt="I love to" scan_layers=false load_parameters_path=$CHECKPOINT_TPU_UNSCANNED

# Output from huggingface 8B model on MaxText:
#Input `I love to` -> ` travel and explore new places, but I also love to stay at home and relax. I'm a bit of a homebody, and I enjoy spending time with my family and friends. I'm a bit of a foodie, and I love trying new recipes and experimenting with different flavors and ingredients. I'm also a bit of a movie buff, and I love watching classic films and new releases alike.
# I'm a bit of a hopeless romantic, and I believe in the idea of true love. I'm looking for someone who shares my values and my sense of humor, and who is always up for a good time. I'm a bit of a goofball, and I love to laugh and have fun. I'm looking for someone who can keep up with me and appreciate my quirks.
# I'm a bit of a creative person, and I love to express myself through art, music, and writing. I'm a bit of a dreamer, and I love to imagine new possibilities and scenarios. I'm a bit of a perfectionist, and I always strive to do my best and be my best self.
# ...

# Output from huggingface DeekSeek 8B model on MaxText:
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
