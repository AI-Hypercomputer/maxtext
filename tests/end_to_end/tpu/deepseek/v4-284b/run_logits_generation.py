import sys
import os

# Ensure maxtext is in path
sys.path.insert(0, os.getcwd())

from tests.assets.logits_generation.generate_hf_golden_logits import save_golden_logits

def main():
  prompt = ("The development of large language models has seen rapid advancements in both architectural design and scaling laws. "
            "As researchers push the boundaries of model size and context length, traditional attention mechanisms face "
            "significant bottlenecks due to their quadratic computational complexity with respect to sequence length. To address this, "
            "novel approaches such as Compressed Sparse Attention and Heterogeneous Compressed Attention have been introduced. "
            "These mechanisms aim to reduce the memory footprint of the Key-Value cache by pooling representations over discrete windows, "
            "thereby enabling much longer sequences to be processed efficiently. In the DeepSeek V4 architecture, this is achieved by "
            "alternating between different compression ratios across the decoder layers. The early prefix layers utilize static hash routing "
            "to distribute tokens deterministically among experts, establishing a strong semantic foundation without the overhead of "
            "learned routing mechanisms. Following the prefix layers, the model employs a scannable block structure that perfectly "
            "alternates between Heavily Compressed Attention, which utilizes a compression ratio of one hundred and twenty-eight, and "
            "Compressed Sparse Attention, which utilizes a compression ratio of four. The Heavily Compressed Attention layers drastically "
            "reduce the sequence length by applying overlapping pooling windows. This means that for every one hundred and twenty-eight "
            "tokens, only a single compressed token representation is stored in the cache. This aggressive compression is particularly "
            "effective for capturing long-range, high-level structural dependencies within the text while keeping the memory footprint "
            "minimal. On the other hand, the Compressed Sparse Attention layers use a much milder compression ratio of four. However, to "
            "maintain efficiency, these layers incorporate a sparse indexer module. The indexer evaluates the query representations "
            "against the compressed windows and selects only the top-k most relevant blocks to attend to. By masking out the irrelevant "
            "blocks, the model avoids computing attention scores across the entire sequence, focusing its computational resources only on "
            "the most critical information. This hybrid approach allows the model to scale to hundreds of billions of parameters and handle "
            "enormous context windows natively. During distributed training on TPU clusters, this architecture requires careful management "
            "of tensor parallelism and expert parallelism to prevent memory fragmentation and communication bottlenecks. The integration "
            "of MoE with these advanced attention mechanisms ensures that the model not only learns diverse features across its experts but "
            "also balances the computational load evenly across the accelerators. As we continue to refine these techniques, the path "
            "toward even more capable and efficient language models becomes increasingly clear, paving the way for applications that can "
            "understand and generate text with unprecedented depth and coherence over massive contexts.")

  # Duplicate 5 times
  duplicated_prompt = " ".join([prompt] * 5)

  print("Generating golden logits using deepseek4-tiny...")
  save_golden_logits(
      model_id="deepseek-ai/DeepSeek-V4-Flash",
      output_path="golden_data_deepseek4-tiny.jsonl",
      prompt_texts=[duplicated_prompt],
      apply_chat_template=False,
      gcs_bucket="snehalv-data",
      hf_model_path="tests/end_to_end/tpu/deepseek/v4-284b/hf_tiny_model",
      hf_load_dtype="bfloat16",
      trust_remote_code=True,
      image_paths=None,
      output_format="json",
  )
  print("Done!")

if __name__ == "__main__":
  main()
