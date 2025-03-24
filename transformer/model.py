import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")


class Tokenizer:
    def __init__(
        self, tokenizer: tiktoken.Encoding = tokenizer, device: torch.device = device
    ):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.n_vocab
        self.device = device

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(text) for text in texts]

    def decode_batch(self, token_ids_list: list[list[int]]) -> list[str]:
        return [self.decode(token_ids) for token_ids in token_ids_list]


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, device: torch.device = device):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(
            num_embeddings=Tokenizer().vocab_size, embedding_dim=d_model, device=device
        )
        self.d_model = d_model

    def forward(self, input_text: str | list[str]) -> tuple[list[int], torch.Tensor]:
        """
        Args:
            input_text: string (for single string input) or
            list of strings (for batch input)

        Returns:
            token_ids: list of integers, the token ids of the input text
            embeddings: tensor, the embeddings of the input text
        """

        assert isinstance(input_text, str) or isinstance(
            input_text, list
        ), f"Input text must be a string (for single string input) or a list of strings (for batch input), received input type: {type(input_text)}"

        if isinstance(input_text, str):
            token_ids = Tokenizer().encode(input_text)
            embeddings = self.embedding(
                torch.stack([torch.tensor(token_ids, device=self.device)], dim=0)
            )
            embeddings = embeddings * math.sqrt(self.d_model)
            return token_ids, embeddings

        elif isinstance(input_text, list):
            token_ids_list = Tokenizer().encode_batch(input_text)
            max_seq_len = max([len(token_ids) for token_ids in token_ids_list])
            padded_token_ids_list = []
            for token_ids in token_ids_list:
                # basically, we're padding the token ids to the max sequence length by
                # adding 0s to the end of the token ids list until it's the same length
                # as the longest token ids list in the batch
                padded_token_ids = token_ids + [0] * (max_seq_len - len(token_ids))
                padded_token_ids_list.append(padded_token_ids)
            embeddings_list = [
                self.embedding(
                    torch.stack([torch.tensor(token_ids, device=self.device)], dim=0)
                )
                for token_ids in padded_token_ids_list
            ]
            embeddings_list = [
                embeddings * math.sqrt(self.d_model) for embeddings in embeddings_list
            ]
            return padded_token_ids_list, embeddings_list


"""
Original positional encoding from the paper. Just for reference.

class PositionalEncodingOriginalPaper(nn.Module):
    def __init__(self, device: Optional[torch.device] = device):
        super().__init__()
        self.device = device

    def forward(self, input_embeddings):
        batch, seq_len, d_model = input_embeddings.shape

        pos = torch.arange(seq_len, dtype=torch.float32, device=self.device).unsqueeze(1)
        dim = torch.arange(d_model, dtype=torch.float32, device=self.device)

        angle_rates = pos / (10000 ** (2 * dim / d_model)
        pe = torch.zeros(seq_len, d_model, device=self.device)
        pe[:, 0::2] = torch.sin(angle_rates[:, 0::2])
        pe[:, 1::2] = torch.cos(angle_rates[:, 1::2])

        return pe.unsqueeze(0).expand(batch, -1, -1) + input_embeddings
"""


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the input embeddings.

    Args:
        device: Optional[torch.device], the device to run the
                positional encoding on
        dropout: float, the dropout rate for the positional encoding
    """

    def __init__(self, dropout: float, device: torch.device = device):
        super().__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_embeddings):
        """
        Currently works for single string input.

        TODO:
            - Make it work for batch inputF
        """
        batch, seq_len, d_model = input_embeddings.shape
        pos = torch.arange(seq_len, device=self.device).unsqueeze(1)
        dim = torch.arange(d_model, device=self.device)

        angle_rates = pos * torch.exp(
            (-2 * dim * torch.log(torch.tensor(10000, device=self.device))) / d_model
        )

        pe = torch.zeros(seq_len, d_model, device=self.device)
        pe[:, 0::2] = torch.sin(angle_rates[:, 0::2])
        pe[:, 1::2] = torch.cos(angle_rates[:, 1::2])

        return self.dropout(pe.unsqueeze(0).expand(batch, -1, -1) + input_embeddings)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        mask: torch.Tensor | None,
        dropout: float,
        device: torch.device = device,
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
        self.device = device
        assert (
            d_model % num_heads == 0
        ), f"""
Model dim is not divisible by num_heads. Please ensure that
the division is possible.
Model dim: {d_model}, Number of heads: {num_heads}"""

        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, device=device)
        """
        Shape: (seq_length, d_model) -> (seq_length, d_model)
        """

        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, device=device)
        """
        Shape: (seq_length, d_model) -> (seq_length, d_model)
        """

        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, device=device)
        """
        Shape: (seq_length, d_model) -> (seq_length, d_model)
        """

        self.W_o = nn.Linear(
            in_features=num_heads * self.d_k, out_features=d_model, device=device
        )
        """
        Used to project the concatenated context vectors back to
        the model dimension.

        Shape: (num_heads * d_k, d_model). Or simply (d_model, d_model).
        The original paper uses the term d_v instead of d_k, but d_v is the
        same as d_k.
        """

        self.mask = mask
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def self_attention(
        query, key, value, dropout: nn.Dropout, mask=None, device: torch.device = device
    ):
        _, _, _, d_k = query.shape

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            if mask.dim() == 2:
                # From [seq_len_q, seq_len_k] to [1, 1, seq_len_q, seq_len_k]
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                # From [1, seq_len_q, seq_len_k] to [1, 1, seq_len_q, seq_len_k]
                mask = mask.unsqueeze(1)

            mask = mask.to(device)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        if dropout is not None:
            attn_weights = dropout(attn_weights)
        output = torch.matmul(attn_weights, value)

        return output, attn_weights

    def forward(self, q_encodings, k_encodings, v_encodings) -> torch.Tensor:
        # (batch, seq_length, d_model) -> (batch, seq_length, d_model)
        q = self.W_q(q_encodings)

        # (batch, seq_length, d_model) -> (batch, seq_length, d_model)
        k = self.W_k(k_encodings)

        # (batch, seq_length, d_model) -> (batch, seq_length, d_model)
        v = self.W_v(v_encodings)

        query = q.view(q.shape[0], q.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        # ========================== ↑ Query Tensor Reshape Logic ↑ ==========================
        # (batch, seq_length, d_model) {view} -> (batch, seq_length, num_heads, d_k) {transpose} -> (batch, num_heads, seq_length, d_k)

        key = k.view(k.shape[0], k.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = v.view(v.shape[0], v.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        # shape of output => (batch, num_heads, seq_length, d_k)
        output, self.attn_weights = MultiHeadAttention.self_attention(
            query=query,
            key=key,
            value=value,
            dropout=self.dropout,
            mask=self.mask,
            device=self.device,
        )

        H = torch.transpose(output, 1, 2).contiguous()
        H = H.view(H.shape[0], -1, H.shape[-1] * H.shape[-2])
        # ========================== ↑ H (concatenated head) Tensor Logic ↑ ==========================
        # (batch, num_heads, seq_length, d_k) {transpose} -> (batch, seq_length, num_heads, d_k) {contiguous} -> (batch, seq_length, num_heads * d_k)

        return self.W_o(H)


class ResidualConnection(nn.Module):
    """
    Residual connection for the encoder/decoder block.

    Mentioned in the original paper "Attention is All You Need" on
    page 3, section 3.1: "Encoder and Decoder Stacks"

    Formula:
    LayerNorm(x + Dropout(Sublayer(x)))

    Note:
        We are calling this class the "Residual Connection"; however, the output of this 
        is comprised of two components:
        
            1. The residual connection (x + Dropout(Sublayer(x)))
            2. The layer normalization (LayerNorm(residual_connection))
    """

    def __init__(self, droput: float, device: torch.device = device):
        super().__init__()
        self.device = device
        self.norm = LayerNorm()
        self.dropout = nn.Dropout(droput)

    def forward(self, x: torch.Tensor, sublayer: nn.Module):
        """
        Args:
            x: tensor, the input to the sublayer
            sublayer: the sublayer of the encoder/decoder block, i.e, either the
            multi-head attention mechanism or the feed-forward network.

        Returns:
            tensor, the output of the sublayer
        """
        sublayer_output = sublayer(x)
        dropped_output = self.dropout(sublayer_output)
        residual_output = x + dropped_output
        normalization = self.norm(residual_output) 
        return normalization


class LayerNorm(nn.Module):
    """
    Layer normalization for the encoder/decoder block.

    Calculated as follows:
    LayerNorm(x) = γ * (x - E[x]) / sqrt(Var[x] + ε) + β

    Where:
        - γ (gamma): scale, i.e., the multiplicative factor
        - β (beta): shift, i.e., the additive factor
        - E[x]: mean of x
        - Var[x]: variance of x
        - ε: small constant to avoid division by zero

    Mentioned in the original paper "Attention is All You Need" on
    page 3, section 3.1: "Encoder and Decoder Stacks". Originally
    proposed by Ba, Kiros, and Hinton in "Layer Normalization" (2016):
    https://arxiv.org/abs/1607.06450

    Formula in mathematical terms:
    LayerNorm(x) = (x - E[x]) / sqrt(Var[x] + ε)

    Additionally, 2 learnable parameters are used to scale and shift the
    normalized output:
        - gamma (γ): scale, i.e., the multiplicative factor
        - beta (β): shift, i.e., the additive factor

    The formula then becomes:
    LayerNorm(x) = γ * (x - E[x]) / sqrt(Var[x] + ε) + β

    Key reasons for adding gamma and beta:
        - To preserve the model's ability to learn complex patterns despite normalization
        - To allow different layers to develop unique feature scaling profiles
        - To provide a controlled "reset" capability - network can learn
            to disable normalization (γ→1, β→0) if needed
        - To compensate for potential information loss during standardization
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.epsilon = 1e-5
        # Learnable parameter for scale
        self.gamma = nn.Parameter(torch.ones(d_model))
        # Learnable parameter for shift
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean_x = torch.mean(x, dim=-1, keepdim=True)
        # unbiased = False means dividing by `n` and not `n-1`
        var_x = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        normalized_x = (x - mean_x) / math.sqrt(var_x + self.epsilon)
        return self.gamma * normalized_x + self.beta


class FeedForward(nn.Module):
    """
    Feed forward network for the encoder/decoder block.

    Args:
        d_model: integer, the dimension of the model
        d_ff: integer, the dimension of the feed forward network's inner layer.
                Should be greater than d_model.
        dropout: float, the dropout rate for the feed forward network

    Returns:
        tensor, the output of the feed forward network

    Mentioned in the original paper "Attention is All You Need" on
    page 5, section 3.3: "Position-wise Feed-Forward Networks":

    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

    where:
        - x: input tensor
        - W_1: weight matrix for the first linear layer
        - b_1: bias for the first linear layer
        - W_2: weight matrix for the second linear layer
        - b_2: bias for the second linear layer

    The paper states that the dimensionality of input and output is
    d_model = 512, and the inner-layer has dimensionality d_ff = 2048.
    """

    def __init__(
        self, d_model: int, d_ff: int, dropout: float, device: torch.device = device
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.device = device

        self.linear_1 = nn.Linear(in_features=d_model, out_features=d_ff, device=device)
        self.linear_2 = nn.Linear(in_features=d_ff, out_features=d_model, device=device)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear_2(F.relu(self.linear_1(x))))


class Encoder(nn.Module):
    """
    Transformer encoder.

    Current args (incomplete):
        vocab_size: integer, the size of the vocabulary of the tokenizer
        d_model: integer, the dimension of the model
        num_heads: integer, the number of attention heads
        seq_length: integer, the length of the sequence (any input text length)
        dropout: float, the dropout rate for the attention scores
        causal_masking: boolean, whether to use causal masking

    Returns:
        output: tensor, the output of the transformer encoder
        shape: (batch_size, seq_length, d_model)

    To add:
        Feed forward
        Add & Norm (LayerNorm)
        Residual connection
    """

    def __init__(self, config: dict, device: torch.device = device):
        super().__init__()
        self.config = config
        self.device = device

        self.input_embedding = InputEmbedding(
            d_model=self.d_model(), device=self.device
        )
        self.positional_encoding = PositionalEncoding(
            dropout=self.dropout(), device=self.device
        )

    def forward(self, input_text):
        token_ids, embeddings = self.input_embedding(input_text)
        self.seq_length = len(token_ids)
        q_encodings = self.positional_encoding(embeddings)
        k_encodings = self.positional_encoding(embeddings)
        v_encodings = self.positional_encoding(embeddings)
        return self.multi_head_attention(q_encodings, k_encodings, v_encodings)

    def seq_length(self):
        """
        Return the seq_length attribute if it exists, otherwise return None
        """
        return getattr(self, "seq_length", None)

    def vocab_size(self):
        return Tokenizer().vocab_size

    def d_model(self):
        return self.config["d_model"]

    def num_heads(self):
        return self.config["num_heads"]

    def dropout(self):
        return self.config["dropout"]

    def causal_masking(self):
        return self.config["causal_masking"]

    def create_causal_mask(
        self,
        seq_len_q: Optional[int] = None,
        seq_len_k: Optional[int] = None,
        device: Optional[torch.device] = device,
    ):
        """
        Create a causal mask for attention.

        Args:
            seq_len_q: Query sequence length
            seq_len_k: Key sequence length
            device: Device to create mask on

        Returns:
            Boolean mask where True indicates positions to mask out
        """
        if self.causal_masking():
            mask = torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=device)
            mask = torch.triu(mask, diagonal=1)
            return mask
        else:
            return None

    def multi_head_attention(self, q_encodings, k_encodings, v_encodings):
        return MultiHeadAttention(
            num_heads=self.num_heads(),
            d_model=self.d_model(),
            mask=self.create_causal_mask(self.seq_length, self.seq_length, self.device),
            dropout=self.dropout(),
            device=self.device,
        )(q_encodings, k_encodings, v_encodings)
