import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, embed_dim=2, row_dim=0, col_dim=1) -> None:
        super().__init__()
        self.W_q = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        self.W_k = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        self.W_v = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self, q_encodings, k_encodings, v_encodings, mask=None) -> torch.Tensor:
        q = self.W_q(q_encodings)
        k = self.W_k(k_encodings)
        v = self.W_v(v_encodings)

        sims = q @ torch.transpose(k, dim0=self.row_dim, dim1=self.col_dim)

        scaled_sims = sims / torch.tensor(k.shape[self.col_dim] ** 0.5)

        if mask is not None:
            # Applying mask by setting masked positions to -inf
            # In softmax, this will ensure those positions get 0 attention
            scaled_sims = scaled_sims.masked_fill(
                mask=mask, value=torch.tensor(float("-inf"))
            )

        attn_weights = torch.softmax(input=scaled_sims, dim=-1)
        attention = attn_weights @ v
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=2, row_dim=0, col_dim=1, num_heads=1) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [Attention(embed_dim, row_dim, col_dim) for _ in range(num_heads)]
        )

        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self, q_encodings, k_encodings, v_encodings, mask=None) -> torch.Tensor:
        return torch.cat(
            [
                head(q_encodings, k_encodings, v_encodings, mask=mask)
                for head in self.heads
            ],
            dim=self.col_dim,
        )


torch.manual_seed(42)

q_encodings = torch.tensor(
    [[1.16, 0.23, 0.12], [0.57, 1.36, 0.23], [4.41, -2.16, 0.12]]
)

k_encodings = torch.tensor(
    [[1.16, 0.23, 0.12], [0.57, 1.36, 0.23], [4.41, -2.16, 0.12]]
)

v_encodings = torch.tensor(
    [[1.16, 0.23, 0.12], [0.57, 1.36, 0.23], [4.41, -2.16, 0.12]]
)

attention = Attention(embed_dim=3, row_dim=0, col_dim=1)

print(
    f"""
Self-attention (since each q, k, v is the same):
{attention(q_encodings, k_encodings, v_encodings)}
"""
)

torch.manual_seed(42)

multihead_attention = MultiHeadAttention(embed_dim=3, row_dim=0, col_dim=1, num_heads=1)

print(
    f"""
Multi-head attention with 1 head (using the same q, k, v,
so has to be the same as self-attention):
{multihead_attention(q_encodings, k_encodings, v_encodings)}
"""
)

torch.manual_seed(42)

multihead_attention2 = MultiHeadAttention(
    embed_dim=3, row_dim=0, col_dim=1, num_heads=2
)

print(
    f"""
Multi-head attention with 2 heads:
{multihead_attention2(q_encodings, k_encodings, v_encodings)}
"""
)

print(
    f"""========================================
Everything above was without the mask. Doing it with a mask from here.
"""
)

# Creating a mask of same size as attention weights (which is q.shape[0] x k.shape[0])
mask = torch.tril(
    torch.ones(q_encodings.shape[0], k_encodings.shape[0], dtype=torch.bool)
)
mask = ~mask  # Inverting to mask future positions (True values will be masked)
print(f"Mask:\n{mask}")

torch.manual_seed(42)


# Creating a class that gives me access to attention weights for visualization
class AttentionWithWeights(Attention):
    """
    This is exactly the same as the forward method in the Attention class,
    except that it returns the attention weights in addition to the attention.

    Only for properly visualizing the attention weights, no other purpose.
    """

    def forward(
        self, q_encodings, k_encodings, v_encodings, mask=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.W_q(q_encodings)
        k = self.W_k(k_encodings)
        v = self.W_v(v_encodings)

        sims = q @ torch.transpose(k, dim0=self.row_dim, dim1=self.col_dim)
        scaled_sims = sims / torch.tensor(k.shape[self.col_dim] ** 0.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(
                mask=mask, value=torch.tensor(float("-inf"))
            )

        attn_weights = torch.softmax(input=scaled_sims, dim=-1)
        attention = attn_weights @ v
        return attention, attn_weights


# Storing results for comparison
torch.manual_seed(42)
unmasked_attn = AttentionWithWeights(row_dim=0, col_dim=1, embed_dim=3)
unmasked_result, unmasked_weights = unmasked_attn(q_encodings, k_encodings, v_encodings)

torch.manual_seed(42)
masked_attn = AttentionWithWeights(row_dim=0, col_dim=1, embed_dim=3)
masked_result, masked_weights = masked_attn(
    q_encodings, k_encodings, v_encodings, mask=mask
)

print(
    f"""========================================
ATTENTION WEIGHTS COMPARISON:

Unmasked attention weights (rows attend to columns):
{unmasked_weights}

Masked attention weights (rows attend to columns):
{masked_weights}

Notice in masked weights:
- Row 0 can only attend to position 0 (weight = 1.0)
- Row 1 can only attend to positions 0 and 1
- Row 2 can attend to all positions (unchanged)

Self-attention WITHOUT mask:
{unmasked_result}

Self-attention WITH mask:
{masked_result}

Difference (masked - unmasked):
{masked_result - unmasked_result}
"""
)
