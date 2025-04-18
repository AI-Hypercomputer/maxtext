import numpy as np

mt_before = np.load("query_before_rope.npy")
mt_after = np.load("query_after_rope.npy")

hf_before = np.load("query_before_rope_hf.npy")
hf_after = np.load("query_after_rope_hf.npy")
mt_after_hf_input = np.load("query_after_rope_same_input.npy")

print(mt_after)
print(mt_after_hf_input)
print(hf_after)



ATOL, RTOL = 1e-3, 1e-3


print(mt_before.shape)
print(mt_after.shape)
print(hf_before.shape)
print(hf_after.shape)

# np.testing.assert_allclose(mt_before, hf_before, rtol=RTOL, atol=ATOL)
# np.testing.assert_allclose(mt_after, hf_after, rtol=RTOL, atol=ATOL)


# # Compute absolute differences
# hf_before = hf_before.reshape(-1)
# mt_before = mt_before.reshape(-1)
# diff = np.abs(hf_before - mt_before)

# # Get indices of top 10 largest differences
# top_indices = np.argsort(-diff)[:10]  # negative sign for descending order

# print(top_indices)

# # Show results
# print("Top 10 differences:")
# for idx in top_indices:
#     print(f"Index {idx}: |{hf_before[idx]} - {mt_before[idx]}| = {diff[idx]}")







# # Compute absolute differences
# hf_after = hf_after.reshape(-1)
# mt_after = mt_after.reshape(-1)
# diff = np.abs(hf_after - mt_after)

# # Get indices of top 10 largest differences
# top_indices = np.argsort(-diff)[:10]  # negative sign for descending order

# print(top_indices)

# # Show results
# print("Top 10 differences:")
# for idx in top_indices:
#     print(f"Index {idx}: |{hf_after[idx]} - {mt_after[idx]}| = {diff[idx]}")