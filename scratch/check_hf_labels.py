from diffusers import DiTPipeline

pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256")

words = ["white shark", "umbrella"]
class_ids = pipe.get_label_ids(words) if hasattr(pipe, 'get_label_ids') else "No get_label_ids method"

print(f"Words: {words}")
print(f"Class IDs: {class_ids}")

# If get_label_ids doesn't exist, let's look at pipe.labels
if hasattr(pipe, 'labels'):
    print(f"Labels type: {type(pipe.labels)}")
    print("Sample:", list(pipe.labels.items())[:5])
    
    # Check if 2 and 879 are in values or keys
    # If pipe.labels is mapping from name to ID or ID to name?
    # Usually it is dict[int, str] or dict[str, int]
    # Let's check keys type
    if len(pipe.labels) > 0:
        sample_key = list(pipe.labels.keys())[0]
        print(f"Sample key type: {type(sample_key)}")
        print(f"Sample value type: {type(list(pipe.labels.values())[0])}")
        
    # Let's see if words are in keys
    for word in words:
        if word in pipe.labels:
            print(f"'{word}' is in keys, value: {pipe.labels[word]}")
        else:
            print(f"'{word}' is NOT in keys")
            
    # Let's see if IDs are in keys
    for cid in [2, 879]:
        if cid in pipe.labels:
            print(f"{cid} is in keys, value: {pipe.labels[cid]}")
        else:
            print(f"{cid} is NOT in keys")

else:
    print("No pipe.labels")
