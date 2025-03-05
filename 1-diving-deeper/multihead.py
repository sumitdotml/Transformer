import torch
import torch.nn as nn
from imports import tokenize_text, positional_encoding

input_text = "I like my coffee"
torch.manual_seed(42)

input_ids, tokenized = tokenize_text(input_text)

# for the sake of encoder where all the q, k & v encodings are the same
q_encodings = positional_encoding(tokenized)
k_encodings = positional_encoding(tokenized)
v_encodings = positional_encoding(tokenized)

print(q_encodings.shape, q_encodings)


class MultiHeadAttentionV2(nn.Module):
    def __init__(self, num_heads:int, d_model:int, seq_length: int, dropout: float) -> None:
        """
        num_heads: Total number of heads for multihead attention
        d_model: Dimension of the model (i.e., the size of the input embedding vector)
        seq_length: Length of the sequence (i.e., the number of tokens in a given input text)
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
        
        self.d_k = d_model // num_heads # d_k for multihead attn
        """
        d_k is the dimension of the key vector for each individual head.
        It is obtained by splitting the d_model dimension into num_heads
        equal parts.
        If d_model = 512 and num_heads = 8, then d_k = 512 / 8 = 64.
        """
        
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model)
        """
        Operation that is happening here: q_encodings -> W_q.
        Shape: (seq_length, d_model) -> (seq_length, d_model)
        """
        
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model)
        """
        Operation that is happening here: k_encodings -> W_k.
        Shape: (seq_length, d_model) -> (seq_length, d_model)
        """
        
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model)
        """
        Operation that is happening here: v_encodings -> W_v.
        Shape: (seq_length, d_model) -> (seq_length, d_model)
        """
        
        self.W_o = nn.Linear(in_features = num_heads * self.d_k, out_features = d_model)
        """This is the `W_o` matrix that is used to project the concatenated
        context vectors back to the model dimension.
        Shape: (num_heads * d_k, d_model). Or simply (d_model, d_model).\n
        The original paper uses the term d_v instead of d_k, but d_v is the
        same as d_k.
        """

    def forward(self, q_encodings, k_encodings, v_encodings) -> torch.Tensor:
        """
        Not writing this with the decoder in mind. Will edit accordingly
        later. I am assuming my input encoded tensor here has a shape of
        [1, 4, 512]. So this is not a batch but rather a single input (or
        I can also say a batch of 1).
        """
        q = self.W_q(q_encodings)
        k = self.W_k(k_encodings)
        v = self.W_v(v_encodings)

        # transposing the last 2 dimensions of k since k is a 3D tensor
        qk_T = q @ torch.transpose(k, -2, -1)
        print("qk_T", qk_T.shape)  # [1, 4, 4]
        attn_scores = qk_T / (k.shape[-1] ** 0.5)
        print("attn_scores (qk_T / sqrt(d_k)):", attn_scores.shape)  # [1, 4, 4]
        attn_weights = torch.softmax(attn_scores, dim=-1)
        print("attn_weights", attn_weights.shape)  # [1, 4, 4]
        context_vec = attn_weights @ v
        print("Output (context_vec):", context_vec.shape)  # [1, 4, 512]
        return context_vec


mulhead = MultiHeadAttentionV2(num_heads=8, d_model=512, context_length=4)

sample_mulhead = mulhead(q_encodings, k_encodings, v_encodings)
print(sample_mulhead)
