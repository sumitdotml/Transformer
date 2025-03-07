import torch
import torch.nn as nn
import math
from typing import Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Transformer(nn.Module):
    """
    Transformer model.

    Current args (incomplete):
        vocab_size: integer, the size of the vocabulary of the tokenizer
        d_model: integer, the dimension of the model
        num_heads: integer, the number of attention heads
        seq_length: integer, the length of the sequence (any input text length)
        dropout: float, the dropout rate for the attention scores
        causal_masking: boolean, whether to use causal masking

    Returns:
        output: tensor, the output of the transformer model
        shape: (batch_size, seq_length, d_model)

    To add:
        Feed forward
        Add & Norm
        Residual connection
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

    def forward(self, input_text):
        token_ids, embeddings = self.tokenize_text(input_text)
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
        return self.config["vocab_size"]
    
    def tokenizer(self):
        return self.config["tokenizer"]
    
    def d_model(self):
        return self.config["d_model"]
    
    def num_heads(self):
        return self.config["num_heads"]
    
    def dropout(self):
        return self.config["dropout"]
    
    def causal_masking(self):
        return self.config["causal_masking"]
    
    def create_causal_mask(self, seq_len_q: Optional[int] = None, seq_len_k: Optional[int] = None, device=None):
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
    
    def embedding_matrix(self):
        return nn.Embedding(num_embeddings=self.vocab_size(), embedding_dim=self.d_model())
    
    def tokenize_text(self, text):
        token_ids = self.tokenizer().encode(text)
        embeddings = self.embedding_matrix()(torch.stack([torch.tensor(token_ids)], dim=0))
        return token_ids, embeddings
    
    def tokenize_batch(self, texts):
        token_ids_list = [self.tokenizer().encode(text) for text in texts]
        max_seq_len = max([len(token_ids) for token_ids in token_ids_list])
        padded_token_ids = []
        for ids in token_ids_list:
            padded_ids = ids + [0] * (max_seq_len - len(ids))
            padded_token_ids.append(padded_ids)
        tokens_tensor = torch.tensor(padded_token_ids)
        embeddings = self.embedding_matrix()(tokens_tensor)
        return token_ids_list, embeddings
    
    def decode_text(self, token_ids):
        return self.tokenizer().decode(token_ids)
    
    def decode_batch(self, token_ids_list):
        return [self.decode_text(token_ids) for token_ids in token_ids_list]
    
    def positional_encoding_original_paper(self, input_embeddings):
        batch, seq_len, d_model = input_embeddings.shape
        device = input_embeddings.device

        pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        dim = torch.arange(d_model, dtype=torch.float32, device=device)

        angle_rates = pos / (10000 ** (2 * dim / d_model))
        
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(angle_rates[:, 0::2])
        pe[:, 1::2] = torch.cos(angle_rates[:, 1::2])

        return pe.unsqueeze(0).expand(batch, -1, -1) + input_embeddings
    
    def positional_encoding(self, input_embeddings):
        batch, seq_len, d_model = input_embeddings.shape
        pos = torch.arange(seq_len).unsqueeze(1)
        dim = torch.arange(d_model)

        angle_rates = pos * torch.exp((-2 * dim * torch.log(torch.tensor(10000))) / d_model)
        
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(angle_rates[:, 0::2])
        pe[:, 1::2] = torch.cos(angle_rates[:, 1::2])

        return pe.unsqueeze(0).expand(batch, -1, -1) + input_embeddings
    
    def multi_head_attention(self, q_encodings, k_encodings, v_encodings):
        return MultiHeadAttention(
            num_heads=self.num_heads(),
            d_model=self.d_model(),
            mask=self.create_causal_mask(self.seq_length, self.seq_length),
            dropout=self.dropout(),
        )(q_encodings, k_encodings, v_encodings)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        mask: torch.Tensor | None,
        dropout: float,
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
        _, _, _, d_k = query.shape
        _, _, seq_len_k, _ = key.shape
        
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            if mask.dim() == 2:
                # From [seq_len_q, seq_len_k] to [1, 1, seq_len_q, seq_len_k]
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                # From [1, seq_len_q, seq_len_k] to [1, 1, seq_len_q, seq_len_k]
                mask = mask.unsqueeze(1)
                
            mask = mask.to(device)  # Ensure same device
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        
        # Rest of the function (softmax, dropout, etc.)
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

        query = q.view(q.shape[0], q.shape[1],
                       self.num_heads, self.d_k).transpose(1, 2)
        # ========================== ↑ Query Tensor Reshape Logic ↑ ==========================
        # (batch, seq_length, d_model) {view} -> (batch, seq_length, num_heads, d_k) {transpose} -> (batch, num_heads, seq_length, d_k)

        key = k.view(k.shape[0], k.shape[1],
                     self.num_heads, self.d_k).transpose(1, 2)
        value = v.view(v.shape[0], v.shape[1],
                       self.num_heads, self.d_k).transpose(1, 2)

        # shape of output => (batch, num_heads, seq_length, d_k)
        output, self.attn_weights = MultiHeadAttention.self_attention(
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