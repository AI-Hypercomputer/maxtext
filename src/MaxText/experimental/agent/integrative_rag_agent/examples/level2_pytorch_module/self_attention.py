import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "embed_size needs to be divisible by heads"

        # FIX: Input to Linear layers is embed_size, not head_dim
        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # FIX: Get the batch size (N) and sequence lengths correctly
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Project the input embeddings
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # Split the embedding into self.heads pieces
        # Reshape: (N, Seq_Len, Heads, Head_Dim)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # Transpose to bring Heads dimension forward for parallel multiplication
        # New shape: (N, Heads, Seq_Len, Head_Dim)
        values = values.transpose(1, 2)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # Calculate Energy (Scaled Dot-Product Attention)
        # Shape: (N, Heads, Query_Len, Head_Dim) * (N, Heads, Head_Dim, Key_Len) 
        # Result: (N, Heads, Query_Len, Key_Len)
        energy = torch.matmul(queries, keys.transpose(-2, -1))

        # Scale the energy
        energy = energy / (self.head_dim ** 0.5)

        # Apply Mask (if provided)
        if mask is not None:
            # mask shape should broadcast to (N, 1, 1, Key_Len) or similar
            # We fill masked elements with a very low value so Softmax makes them 0
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy to probabilities
        attention = torch.softmax(energy, dim=3)

        # Apply attention to values
        # (N, Heads, Query_Len, Key_Len) * (N, Heads, Value_Len, Head_Dim) 
        # Result: (N, Heads, Query_Len, Head_Dim)
        out = torch.matmul(attention, values)

        # Reshape back to original dimensions
        # Swap Heads and Query_Len back: (N, Query_Len, Heads, Head_Dim)
        out = out.transpose(1, 2).contiguous()
        
        # Flatten the last two dimensions: (N, Query_Len, Embed_Size)
        out = out.reshape(N, query_len, self.heads * self.head_dim)

        # Final linear layer
        out = self.fc_out(out)

        return out
