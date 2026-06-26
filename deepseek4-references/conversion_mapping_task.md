### Task
Take a look at both @maxtext/deepseek4-references/huggingface_structure.txt and @maxtext/deepseek4-references/maxtext_structure.txt. from those i want you to generate a text/json/python/csv/{other format} file of your choosing that takes each line in the maxtext structure file and maps it to one or multiple parameters in the huggingface structure file. Below I will share some hints and reference files which you should use to guide your mapping. This mapping is meant for me to review and then we will later proceed to use it to convert the checkpoint after I manually go through and verify it.

#### Hints
There is repetition to the layer structure. 
- The first two layers are identitcal in structure. 
- The third layer will be a compressed sparse attention layer as described in @maxtext/deepseek4-references/huggingface-pytorch-references/deepseek_v4_paper.tex.
- The layers after the third layer will be repeating pattern of 1 hyper compressed attention layer and 1 compressed sparse attention layer.
- Although the third layer is compressed sparse attention similar to the layer at 2*layer_index + 2, it's not exactly the same so we will require a seperate mapping for both, don't try to share the same mapping for them. 
- Overall we need mappings for the the first two layers, the third layer, and the repeating pattern of the layers after that. In this case the total number of layers is 7. So the layers are 0, 1, 2, 3, 4, 5, 6. Layer 0 and 1 are the first two layers, layer 2 is the third layer, and layers 3, 4, 5, 6 are the repeating pattern of 1 hyper compressed attention layer and 1 compressed sparse attention layer. So layer 3 is hyper compressed attention, layer 4 is compressed sparse attention, layer 5 is hyper compressed attention, and layer 6 is compressed sparse attention.

#### References
- Paper: @maxtext/deepseek4-references/huggingface-pytorch-references/deepseek_v4_paper.tex
- Huggingface implementation: @maxtext/deepseek4-references/huggingface-pytorch-references/modeling_deepseek_v4.py
- Maxtext implementation: @maxtext/src/maxtext/layers/attention_compressed.py. @maxtext/src/maxtext/layers/mhc.py. @maxtext/src/maxtext/models/deepseek4.py. @maxtext/src/maxtext/layers/mhc.py. @maxtext/src/maxtext/models/deepseek.py. The unit tests in @maxtext/tests/unit/deepseek_v4_vs_reference_test.py could be especially helpful because they show how we reformat huggingface reference model weights to maxtext to test parity. @maxtext/src/maxtext/layers/moe.py.