import transformers
from transformers import Llama4ForConditionalGeneration, AutoConfig

config = AutoConfig.from_pretrained("/home/ranran/config_4layer.json")
model = Llama4ForConditionalGeneration(config=config)

model.save_pretrained("/home/ranran/maverick_4layer")
for name, val in model.named_parameters():
  print(name, val.shape)