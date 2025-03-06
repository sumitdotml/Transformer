"""
To the future me: There's some nice mental model I could make of the changes
in the shape of the tensors in the multihead attention mechanism in the
shape_study_notebook.ipynb file in the same directory as this file. Refer to
that and everything should make sense.
"""

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

print(
    f"\nq_encodings.shape: {q_encodings.shape}\nq_encodings: {
      q_encodings.shape[0]}\n"
)
print(
    f"k_encodings.shape: {k_encodings.shape}\nk_encodings: {
      k_encodings.shape[0]}\n"
)
print(
    f"v_encodings.shape: {v_encodings.shape}\nv_encodings: {
      v_encodings.shape[0]}\n"
)


class MultiHeadAttentionV2(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        seq_length: int = 4,
        dropout: float = 0.1,
    ) -> None:
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

        self.d_k = d_model // num_heads  # d_k for multihead attn
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

        self.W_o = nn.Linear(in_features=num_heads * self.d_k, out_features=d_model)
        """This is the `W_o` matrix that is used to project the concatenated
        context vectors back to the model dimension.
        Shape: (num_heads * d_k, d_model). Or simply (d_model, d_model).\n
        The original paper uses the term d_v instead of d_k, but d_v is the
        same as d_k.
        """

        self.dropout = nn.Dropout(dropout)

    def forward(self, q_encodings, k_encodings, v_encodings) -> torch.Tensor:
        """
        Not writing this with the decoder in mind. Will edit accordingly
        later. I am assuming my input encoded tensor here has a shape of
        [1, 4, 512]. So this is not a batch but rather a single input (or
        I can also say a batch of 1).
        """

        # (batch, seq_length, d_model) -> (batch, seq_length, d_model)
        q = self.W_q(q_encodings)
        print("q's size after q_encodings @ W_q:", q.shape)

        # (batch, seq_length, d_model) -> (batch, seq_length, d_model)
        k = self.W_k(k_encodings)

        # (batch, seq_length, d_model) -> (batch, seq_length, d_model)
        v = self.W_v(v_encodings)

        query = q.view(q.shape[0], q.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        """
        Here, I split the q tensor with shape (batch, seq_length, d_model) into 
        num_heads and then transposed the last 2 dimensions to get a shape of 
        (batch, num_heads, seq_length, d_k).

        In other words, the flow is:
        (batch, seq_length, d_model) {view} -> (batch, seq_length, num_heads, d_k) {transpose} -> (batch, num_heads, seq_length, d_k)

        **Why is the transpose necessary?**
        - If I use the logic of a simple self-attention, I know that the seq_length
        and d_model are always the last 2 dimensions of the tensor.
        - So, if I want to split the tensor into num_heads parts, I need to reverse the order
        of num_heads and seq_length
        - After the transpose, the shape of the tensor is 
        (batch, num_heads, seq_length, d_k).
        - Another reason is that the transpose operation brings the head
        dimension to the front, which allows for easier computation of attention scores.

        This shape is obtained by:
        - view: Splitting the tensor into num_heads parts.
        - transpose: Transposing the last 2 dimensions to bring the head dimension
        to the front.
        """

        key = k.view(k.shape[0], k.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        """
        Operation flow of the key tensor:
        (batch, seq_length, d_model) {view} -> (batch, seq_length, num_heads, d_k) {transpose} -> (batch, num_heads, seq_length, d_k)
        """

        value = v.view(v.shape[0], v.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        """
        Operation flow of the value tensor:
        (batch, seq_length, d_model) {view} -> (batch, seq_length, num_heads, d_k) {transpose} -> (batch, num_heads, seq_length, d_k)
        """

        print(
            f"\nq after splitting and transposing: {query.shape}"
            f"\nk after splitting and transposing: {key.shape} what"
            f"\nvalue after splitting and transposing: {value.shape}"
        )

        @staticmethod
        def self_attention(query, key, value, mask=False, dropout=None):
            d_k = key.shape[-1]
            attn_scores = (query @ torch.transpose(key, -2, -1)) / d_k**0.5
            if mask:
                mask = torch.unsqueeze(
                    ~torch.tril(
                        torch.ones(query.shape[-2], key.shape[-2], dtype=torch.bool)
                    ),
                    dim=0,
                )
                attn_scores = torch.masked_fill(
                    attn_scores, mask=mask, value=torch.tensor(float("-inf"))
                )

            if dropout is not None:
                attn_scores = self.dropout(attn_scores)

            attn_weights = torch.softmax(attn_scores, dim=-1)
            context_vec = attn_weights @ value
            return context_vec, attn_weights


mulhead = MultiHeadAttentionV2(num_heads=8, d_model=512, seq_length=4)

sample_mulhead = mulhead(q_encodings, k_encodings, v_encodings)
# print(sample_mulhead)


"""
Below are some functions that I created to understand the multihead attention mechanism better.
Not related to the above code.
"""


def calculate_q_for_shape_understanding(batch_size=1, seq_length=4, d_model=512):
    """
    I say q here, but it's the same for k and v as well.
    I'm doing this to see how tensor shapes change from q_encodings -> q when
    doing self.W_q(q_encodings) in the MultiHeadAttention class.
    """
    q_encodings = torch.randn(batch_size, seq_length, d_model)

    # similar to the W_q I created with nn.Embedding in the MultiHeadAttention class
    W_q = torch.randn(d_model, d_model)
    result = q_encodings @ W_q
    print(
        "\n======================== Just experimenting with the W_q matrix =========================="
    )
    print(f"q_encodings @ W_q shape: {result.shape}\nresult: {result}\n")
    return result


def masking_understanding(
    query=torch.randn(1, 4, 512),
    key=torch.randn(1, 4, 512),
    value=torch.randn(1, 4, 512),
    mask=True,
):
    d_k = key.shape[-1]
    attn_scores = (query @ torch.transpose(key, -2, -1)) / d_k**0.5
    if mask:
        mask = torch.unsqueeze(
            ~torch.tril(torch.ones(query.shape[-2], key.shape[-2], dtype=torch.bool)),
            dim=0,
        )
        attn_scores = torch.masked_fill(
            attn_scores, mask=mask, value=torch.tensor(float("-inf"))
        )
    print(
        "=========================== Masking Understanding ==================================="
    )
    print(
        f"attn_scores.shape: {
          attn_scores.shape}\nattn_scores:\n{attn_scores}"
    )
    print(f"torch.softmax(attn_scores, dim=-1):\n{torch.softmax(attn_scores, dim=-1)}")
    print(f"shape after softmax: {torch.softmax(attn_scores, dim=-1).shape}")
    return attn_scores
