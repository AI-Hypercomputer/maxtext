import json

CUTOFF_INPUT = 1024
CUTOFF_OUTPUT = 1024

prefill_bucket_size_to_ms = {128 : 11.60, 256: 15.72, 512 : 25.35, 1024: 46.35} #UPDATE(??)
system_time_per_decode_token_ms = 33.47/96 #UPDATE(IS IT TRUE?)

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def tokens_in_input_str(s):
    return_val =  int(1.3 * len(s.split()))
    #print(f"{s=} -> {return_val=}")
    return return_val

convo_numbers = []
loaded_share_gpt = json.load(open('/home/rwitten/ShareGPT_V3_unfiltered_cleaned_split.json', 'r'))
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

total_prefill_system_ms = 0
total_generate_system_ms = 0

total_system_output_tokens = 0
for convo in kept_convos:
    input_tok, output_tok = convo
    bucket = max(128,next_power_of_2(input_tok))
    generate_system_ms = output_tok * system_time_per_decode_token_ms 
    prefill_system_ms = prefill_bucket_size_to_ms[bucket]

    print(f"{convo=} {bucket=}, {prefill_system_ms=:.2f}, {generate_system_ms=:.2f}")

    total_prefill_system_ms += prefill_system_ms
    total_generate_system_ms += generate_system_ms

total_time_ms = total_prefill_system_ms + total_generate_system_ms
output_tokens = sum([c[1] for c in kept_convos])
print(f"Output tokens {output_tokens} in {total_time_ms/1000:.2f} seconds, for {output_tokens/(total_time_ms/1000):.2f} out tok/s")

total_prefill_sec = total_prefill_system_ms/1000
total_generate_sec = total_generate_system_ms/1000

print(f"Total time {total_time_ms/1000:.2f} seconds, split {total_prefill_sec=:.2f} seconds and {total_generate_sec=:.2f} seconds")