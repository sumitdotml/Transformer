"""
This is the exact same code as the 2-multihead.py file, but without all
the long comments and the print statements I used to understand the
concepts.
"""

import torch
import torch.nn as nn
from imports import tokenize_text, positional_encoding
import math

input_text = "I like my coffee"
torch.manual_seed(42)
device = torch.device("cpu")

input_ids, tokenized = tokenize_text(input_text)

q_encodings = positional_encoding(tokenized)
k_encodings = positional_encoding(tokenized)
v_encodings = positional_encoding(tokenized)


def create_causal_mask(seq_len_q, seq_len_k, device=None):
    """
    Create a causal mask for attention.

    Args:
        seq_len_q: Query sequence length
        seq_len_k: Key sequence length
        device: Device to create mask on

    Returns:
        Boolean mask where True indicates positions to mask out
    """
    mask = torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


class MultiHeadAttentionV2(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        mask: torch.Tensor | None,
        seq_length: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            num_heads: Total number of heads for multihead attention
            d_model: Dimension of the model (i.e., the size of the input embedding vector)
            mask: Mask to apply to the attention scores
            seq_length: Length of the sequence (i.e., the number of tokens in a given input text)
            dropout: Dropout rate for the attention scores
        """
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert (
            d_model % num_heads == 0
        ), f"""
Model dim is not divisible by num_heads. Please ensure that
the division is possible.
Model dim: {d_model}, Number of heads: {num_heads}"""

        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model)
        """
        Shape: (seq_length, d_model) -> (seq_length, d_model)
        """

        self.W_k = nn.Linear(in_features=d_model, out_features=d_model)
        """
        Shape: (seq_length, d_model) -> (seq_length, d_model)
        """

        self.W_v = nn.Linear(in_features=d_model, out_features=d_model)
        """
        Shape: (seq_length, d_model) -> (seq_length, d_model)
        """

        self.W_o = nn.Linear(in_features=num_heads *
                             self.d_k, out_features=d_model)
        """
        Used to project the concatenated context vectors back to the model dimension.
        
        Shape: (num_heads * d_k, d_model). Or simply (d_model, d_model).
        The original paper uses the term d_v instead of d_k, but d_v is the
        same as d_k.
        """

        self.mask = mask
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def self_attention(query, key, value, dropout: nn.Dropout, mask=None, device=None):
        d_k = key.shape[-1]
        attn_scores = (query @ torch.transpose(key, -2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)

        if dropout is not None:
            attn_weights = dropout(attn_weights)
        output = attn_weights @ value
        return output, attn_weights

    def forward(self, q_encodings, k_encodings, v_encodings) -> torch.Tensor:
        # (batch, seq_length, d_model) -> (batch, seq_length, d_model)
        q = self.W_q(q_encodings)
        print("q's size after q_encodings @ W_q:", q.shape)

        # (batch, seq_length, d_model) -> (batch, seq_length, d_model)
        k = self.W_k(k_encodings)

        # (batch, seq_length, d_model) -> (batch, seq_length, d_model)
        v = self.W_v(v_encodings)

        query = q.view(q.shape[0], q.shape[1],
                       self.num_heads, self.d_k).transpose(1, 2)
        # ========================== ↑ Query Tensor Reshape Logic ↑ ==========================
        # (batch, seq_length, d_model) {view} -> (batch, seq_length, num_heads, d_k) {transpose} -> (batch, num_heads, seq_length, d_k)

        key = k.view(k.shape[0], k.shape[1],
                     self.num_heads, self.d_k).transpose(1, 2)
        value = v.view(v.shape[0], v.shape[1],
                       self.num_heads, self.d_k).transpose(1, 2)

        # shape of output => (batch, num_heads, seq_length, d_k)
        output, self.attn_weights = MultiHeadAttentionV2.self_attention(
            query=query,
            key=key,
            value=value,
            dropout=self.dropout,
            mask=self.mask,
            device=device,
        )

        H = torch.transpose(output, 1, 2).contiguous()
        H = H.view(H.shape[0], -1, H.shape[-1] * H.shape[-2])
        # ========================== ↑ H (concatenated head) Tensor Logic ↑ ==========================
        # (batch, num_heads, seq_length, d_k) {transpose} -> (batch, seq_length, num_heads, d_k) {contiguous} -> (batch, seq_length, num_heads * d_k)

        return self.W_o(H)
