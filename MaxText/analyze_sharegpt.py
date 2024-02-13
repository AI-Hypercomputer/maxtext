import json

CUTOFF_INPUT = 1024
CUTOFF_OUTPUT = 1024

def tokens_in_input_str(s):
    return_val =  int(1.3 * len(s.split()))
    #print(f"{s=} -> {return_val=}")
    return return_val

convo_numbers = []
loaded_share_gpt = json.load(open('/home/rwitten/maxtext/ShareGPT_V3_unfiltered_cleaned_split.json', 'r'))
for example in loaded_share_gpt:
    if len(example['conversations']) < 2:
        continue
    input_tokens = tokens_in_input_str(example['conversations'][0]['value'])
    output_tokens = tokens_in_input_str(example['conversations'][1]['value'])
    convo_numbers.append( (input_tokens, output_tokens))

num_convos = len(convo_numbers)
kept_convos = [c for c in convo_numbers if c[0] <= CUTOFF_INPUT and c[1] <= CUTOFF_OUTPUT]

mean_input = sum([c[0] for c in kept_convos]) / len(kept_convos)
mean_output = sum([c[1] for c in kept_convos]) / len(kept_convos)

print(f"Total {num_convos=} but only kept {kept_convos=}. Out of kept, {mean_input=}, {mean_output=}")