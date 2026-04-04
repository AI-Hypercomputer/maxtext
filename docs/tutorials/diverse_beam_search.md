# Tutorial: Diverse Beam Search (DBS) in MaxText

This tutorial explains how to use the **Diverse Beam Search (DBS)** algorithm for generating text in MaxText. DBS is designed to overcome the "lack of diversity" issue in standard beam search by encouraging the model to explore different paths simultaneously.

---

## 1. Overview: Why Diverse Beam Search?
Standard Beam Search often produces very similar sequences (e.g., "The cat sat on the mat", "A cat sat on the mat"). 

**DBS** addresses this by dividing the search into **multiple groups**. Each group is penalized for picking tokens that are already being used by other groups. This results in a much wider variety of high-quality outputs.

## 2. Configuration Parameters
To use DBS in MaxText, you need to configure a few key parameters in your YAML config or as command-line arguments:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `num_beams` | [int](cci:1://file:///Users/kevinwang/Projects/maxtext/src/maxtext/inference/maxengine/maxengine.py:137:2-139:40) | `4` | Total number of beams across ALL groups. |
| `num_groups` | [int](cci:1://file:///Users/kevinwang/Projects/maxtext/src/maxtext/inference/maxengine/maxengine.py:137:2-139:40) | `1` | Number of diversity groups. Must be a divisor of `num_beams`. |
| `diversity_penalty` | `float` | `0.0` | The $\lambda$ parameter. Higher values force more diversity across groups. |

> **Note:** If `num_groups=1`, DBS behaves exactly like standard Beam Search.

## 3. How to Run DBS

### Via Command Line
You can enable DBS by overriding the configuration during the [decode.py](cci:7://file:///Users/kevinwang/Projects/maxtext/src/maxtext/inference/decode.py:0:0-0:0) run:

```bash
python3 -m maxtext.inference.decode \
    config_name=base_model \
    num_beams=8 \
    num_groups=4 \
    diversity_penalty=0.5 \
    prompt="Explain the theory of relativity in simple terms."
```

### Via YAML Configuration
Add the following to your model's configuration file:

```yaml
# dbs_config.yml
num_beams: 8
num_groups: 4
diversity_penalty: 0.5 # Encourages variety between the 4 groups
```

## 4. Best Practices for Choosing Parameters
*   **The Lambda Penalty ($\lambda$):** A value between `0.2` and `0.8` is usually effective. If it's too high, the model might start generating gibberish to "be different." 
*   **Beam vs Group Ratio:** For best results, use at least 2 beams per group (e.g., `num_beams=8` and `num_groups=4`). This allows for local optimization within each group while maintaining global diversity.

## 5. Example Output Comparison

**Standard Beam Search (`num_groups=1`):**
1. "The AI model is very powerful."
2. "This AI model is very powerful."
3. "The AI model is extremely powerful."

**Diverse Beam Search (`num_groups=4`):**
1. "The AI model is very powerful."
2. "Machine learning has revolutionized automation."
3. "Large scale transformers are the state of the art."
4. "Computational efficiency is key for modern LLMs."

---

## 6. Limitations of DBS Implementation
While DBS provides higher quality and more diverse outputs, there are a few important trade-offs to keep in mind:

*   **Higher Memory Requirement**: The size of the **KV Cache** scales linearly with the number of beams. For example, running with `num_beams=8` requires **8x more High-Bandwidth Memory (HBM)** to store the cache compared to standard greedy sampling. Ensure your TPU/GPU has sufficient memory for the beam-expanded batch size.
*   **No Real-time Streaming**: Diverse Beam Search cannot stream tokens one-by-one as they are generated. Because beams can be reordered at any step based on their cumulative scores, the "best" candidate might change mid-sequence. The output is only returned once the entire generation process is complete.

---

## Next Steps
For a deeper dive into the mathematics and architecture of this implementation, check out our [Architecture Overview](./architecture_overview.md) or read the original research paper: [Diverse Beam Search: Decoding Diverse Solutions from Greedy Sequence Models](https://arxiv.org/abs/1610.02424).
