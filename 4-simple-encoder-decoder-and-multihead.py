import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, embed_dim, row_index=0, col_index=1):
        super().__init__()
        self.W_q = nn.Linear(in_features=embed_dim,
                             out_features=embed_dim, bias=False)
        self.W_k = nn.Linear(in_features=embed_dim,
                             out_features=embed_dim, bias=False)
        self.W_v = nn.Linear(in_features=embed_dim,
                             out_features=embed_dim, bias=False)
        self.row_index = row_index
        self.col_index = col_index

    def forward(self, q_encodings, k_encodings, v_encodings, mask=None):
        q = self.W_q(q_encodings)
        k = self.W_k(k_encodings)
        v = self.W_v(v_encodings)

        sims = q @ torch.transpose(k, dim0=self.row_index, dim1=self.col_index)

        scaled_sims = sims / torch.tensor(k.shape[self.col_index] ** 0.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(
                mask, value=torch.tensor(float("-inf"))
            )

        attn_percents = torch.softmax(input=scaled_sims, dim=-1)
        attention = attn_percents @ v
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self):
        pass
