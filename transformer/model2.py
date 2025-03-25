"""
Just me playing around with the help of Gemini.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 0. Determine Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Define Hyperparameters
vocab_size = 10000  # Example vocabulary size
embed_dim = 512
batch_size = 3
max_seq_len = 15
dropout = 0.1
num_heads = 8
ff_dim = 2048
num_layers = 6

# 2. Prepare Input Data (Batch of Texts) - Dummy Data
input_sequence = torch.randint(1, vocab_size, (batch_size, max_seq_len)).to(
    device
)  # (batch_size, seq_len)
print("Input Sequence Shape:", input_sequence.shape)


class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, device):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim).to(device)
        self.embed_dim = embed_dim
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        embedded = self.embedding(x)
        return embedded * math.sqrt(self.embed_dim)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout, max_len=5000, device=None):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.device = device

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe.to(self.device))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, device):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.device = device

        self.W_q = nn.Linear(embed_dim, embed_dim).to(device)
        self.W_k = nn.Linear(embed_dim, embed_dim).to(device)
        self.W_v = nn.Linear(embed_dim, embed_dim).to(device)
        self.W_o = nn.Linear(embed_dim, embed_dim).to(device)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q, K, V: (batch_size, num_heads, seq_len, head_dim)
        # mask (optional): (batch_size, 1, 1, seq_len) - for masking padding

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            mask = mask.to(Q.device)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        output = torch.matmul(attn_probs, V)
        return output

    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (batch_size, seq_len, embed_dim)

        batch_size = Q.size(0)
        seq_len = Q.size(1)

        # Linear projections and split into heads
        Q = (
            self.W_q(Q)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.W_k(K)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.W_v(V)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads and project
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )
        output = self.W_o(attn_output)

        return output


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout, device):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim).to(device)
        self.linear2 = nn.Linear(ff_dim, embed_dim).to(device)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout, device):
        super().__init__()
        self.mha = MultiHeadAttention(embed_dim, num_heads, dropout, device)
        self.ff = FeedForward(embed_dim, ff_dim, dropout, device)
        self.layernorm1 = nn.LayerNorm(embed_dim).to(device)
        self.layernorm2 = nn.LayerNorm(embed_dim).to(device)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x, mask=None):
        # Multi-Head Attention with Residual Connection and Layer Norm
        attn_output = self.mha(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.layernorm1(x)

        # Feed Forward with Residual Connection and Layer Norm
        ff_output = self.ff(x)
        x = x + self.dropout(ff_output)
        x = self.layernorm2(x)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_layers,
        num_heads,
        ff_dim,
        dropout,
        max_len,
        device,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, embed_dim, device)
        self.positional_encoding = PositionalEncoding(
            embed_dim, dropout, max_len, device
        )
        self.encoder_layers = nn.ModuleList(
            [
                EncoderBlock(embed_dim, num_heads, ff_dim, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x, mask=None):
        print("Encoder - Input x shape:", x.shape)

        embedded = self.embedding(x)
        print("Encoder - Embedded shape:", embedded.shape)

        encoded = self.positional_encoding(embedded)
        print("Encoder - Positional Encoding shape:", encoded.shape)

        for i, encoder_layer in enumerate(self.encoder_layers):
            print(f"Encoder - Before Encoder Layer {i+1}, shape: {encoded.shape}")
            encoded = encoder_layer(encoded, mask)
            print(f"Encoder - After Encoder Layer {i+1}, shape: {encoded.shape}")

        print("Encoder - Output shape:", encoded.shape)
        return encoded


# 3. Instantiate Layers
encoder = Encoder(
    vocab_size, embed_dim, num_layers, num_heads, ff_dim, dropout, max_seq_len, device
).to(device)

# 4. Pass through Encoder
encoder_output = encoder(input_sequence)
print("Encoder Output Shape:", encoder_output.shape)
print("Encoder Output:", encoder_output)

"""
Prints:

Using device: cpu
Input Sequence Shape: torch.Size([3, 15])
Encoder - Input x shape: torch.Size([3, 15])
Encoder - Embedded shape: torch.Size([3, 15, 512])
Encoder - Positional Encoding shape: torch.Size([3, 15, 512])
Encoder - Before Encoder Layer 1, shape: torch.Size([3, 15, 512])
Encoder - After Encoder Layer 1, shape: torch.Size([3, 15, 512])
Encoder - Before Encoder Layer 2, shape: torch.Size([3, 15, 512])
Encoder - After Encoder Layer 2, shape: torch.Size([3, 15, 512])
Encoder - Before Encoder Layer 3, shape: torch.Size([3, 15, 512])
Encoder - After Encoder Layer 3, shape: torch.Size([3, 15, 512])
Encoder - Before Encoder Layer 4, shape: torch.Size([3, 15, 512])
Encoder - After Encoder Layer 4, shape: torch.Size([3, 15, 512])
Encoder - Before Encoder Layer 5, shape: torch.Size([3, 15, 512])
Encoder - After Encoder Layer 5, shape: torch.Size([3, 15, 512])
Encoder - Before Encoder Layer 6, shape: torch.Size([3, 15, 512])
Encoder - After Encoder Layer 6, shape: torch.Size([3, 15, 512])
Encoder - Output shape: torch.Size([3, 15, 512])
Encoder Output Shape: torch.Size([3, 15, 512])
Encoder Output: tensor([[[ 1.2075,  0.2035,  1.0337,  ...,  0.2741,  0.3362, -1.3878],
         [ 0.1170, -0.5363, -0.5602,  ..., -0.5526, -1.4579,  0.8263],
         [ 1.0781, -2.0242,  0.0280,  ...,  0.8292, -0.7044, -0.7894],
         ...,
         [ 0.0666, -0.3496,  1.4350,  ..., -0.4715,  0.5515, -0.6433],
         [ 1.4316,  0.1786, -0.4018,  ..., -1.3569,  1.5014, -0.9045],
         [ 0.1907, -1.2621,  0.5232,  ...,  1.5976, -1.1432,  0.5773]],

        [[ 0.0350, -0.3692, -0.0352,  ...,  2.1164, -0.6821, -0.6053],
         [ 3.5586,  0.5181, -0.8149,  ..., -0.4206, -0.7290,  1.6452],
         [ 0.1621,  1.0284, -1.0765,  ...,  1.0532,  0.1141, -1.5833],
         ...,
         [ 0.1873, -0.1464,  0.9092,  ..., -0.4456, -0.1355, -0.3995],
         [ 1.1552, -0.4447, -0.7325,  ...,  1.0162, -0.1951,  1.2546],
         [ 0.7160, -3.3655,  1.2497,  ...,  1.5302, -1.3861,  0.1539]],

        [[-0.4450,  1.0222,  0.7573,  ..., -0.3704,  0.9875, -1.0206],
         [-0.2566, -1.3455,  0.0593,  ...,  0.5640, -0.8519, -1.0486],
         [ 0.6351,  1.0094,  0.8482,  ..., -0.5469, -0.0678, -1.4921],
         ...,
         [-0.3255,  0.9467, -0.5279,  ..., -0.0508, -0.1293, -0.7462],
         [-0.1553, -0.6944,  0.3898,  ...,  1.0271, -2.1448, -1.7609],
         [ 1.2753, -0.4760,  0.0436,  ...,  0.6092, -1.0972, -0.7042]]],
       grad_fn=<NativeLayerNormBackward0>)
"""
