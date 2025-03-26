import math
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
)  # Import from transformers

DEFAULT_TOKENIZER_NAME = "bert-base-uncased"


class InputEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int,
        tokenizer_name: str,
        device: torch.device = device,
        max_length: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.device = device
        self.max_length = max_length  # Store max_length

        # Load the tokenizer
        # Using PreTrainedTokenizerFast for type hinting, common base class
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            tokenizer_name
        )

        # --- Handle Padding Token ---
        if self.tokenizer.pad_token is None:
            print(
                f"Warning: Tokenizer '{tokenizer_name}' does not have a default pad token. Adding '[PAD]'."
            )
            # Common practice: add a pad token if missing, often EOS token is used if available
            # For simplicity here, adding a generic '[PAD]' token.
            # Might need to resize model embeddings if adding tokens to a pre-trained model.
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            # NOTE: If I were fine-tuning a pre-trained model, I'd also need to resize
            # the model's token embeddings matrix: model.resize_token_embeddings(len(tokenizer))

        self.padding_token_id = self.tokenizer.pad_token_id
        self.vocab_size = self.tokenizer.vocab_size  # Get vocab size from HF tokenizer

        print(f"InputEmbedding: Loaded Tokenizer '{tokenizer_name}'")
        print(
            f"InputEmbedding: Vocab Size = {self.vocab_size}, Padding ID = {self.padding_token_id}"
        )

        # Initialize the embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=d_model,
            padding_idx=self.padding_token_id,  # Inform nn.Embedding about the padding index
            device=device,
        )

    def forward(self, input_texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes a batch of input strings into embeddings and a padding mask.

        Args:
            input_texts (List[str]): A list of strings to be processed.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - embeddings (torch.Tensor): A tensor of shape
                  (batch_size, seq_len, d_model) containing the scaled embeddings.
                  `seq_len` is `max_length` if provided, otherwise the length
                  of the longest sequence in the batch.
                - padding_mask (torch.Tensor): A boolean tensor of shape
                  (batch_size, 1, 1, seq_len) where True indicates a position
                  that should be masked (padding). Suitable for attention mechanisms.
        """
        assert isinstance(input_texts, list), "Input must be a list of strings."
        if len(input_texts) < 1:
            assert False, "InputEmbedding: Received an empty list of strings.\n"
        elif len(input_texts) == 1:
            print(f"InputEmbedding: Received {len(input_texts)} string.\n")
        else:
            print(f"InputEmbedding: Received {len(input_texts)} strings.\n")

        # 1. Tokenize the batch using the Hugging Face tokenizer
        # This handles tokenization, adding special tokens, padding, truncation,
        # and tensor conversion in one go.
        tokenizer_args = {
            "text": input_texts,
            "add_special_tokens": True,  # Add [CLS], [SEP] etc. (depends on tokenizer type)
            "padding": (
                "longest" if self.max_length is None else "max_length"
            ),  # Pad to longest in batch or to max_length
            "truncation": self.max_length
            is not None,  # Truncate if max_length is specified
            "max_length": self.max_length,  # Set max_length if provided
            "return_tensors": "pt",  # Return PyTorch tensors
            "return_attention_mask": True,  # Get the attention mask
        }
        encoding = self.tokenizer(**tokenizer_args).to(self.device)

        input_ids_tensor = encoding["input_ids"]  # Shape: (batch_size, seq_len)
        attention_mask_tensor = encoding[
            "attention_mask"
        ]  # Shape: (batch_size, seq_len), 1 for real, 0 for padding

        current_seq_len = input_ids_tensor.shape[1]
        print(f"InputEmbedding: Tokenized IDs tensor shape = {input_ids_tensor.shape}")
        print(
            f"InputEmbedding: Attention mask tensor shape = {attention_mask_tensor.shape}"
        )

        # 2. Create padding mask for attention mechanism
        # I'll need mask to be True where attention should be inhibited (padding tokens)
        # The attention_mask is 0 for padding, 1 for non-padding.
        # So, I'll create the mask where attention_mask == 0.
        # Shape: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
        padding_mask = (attention_mask_tensor == 0).unsqueeze(1).unsqueeze(2)
        print(
            f"InputEmbedding: Padding mask shape for attention = {padding_mask.shape}"
        )

        # 3. Get embeddings (single operation for the whole batch)
        # Shape: (batch_size, seq_len, d_model)
        embeddings = self.embedding(input_ids_tensor)
        print(
            f"InputEmbedding: Embeddings tensor shape (before scaling) = {embeddings.shape}"
        )

        # 4. Scale embeddings (common practice, mentioned in "Attention Is All You Need")
        embeddings = embeddings * math.sqrt(self.d_model)

        return embeddings, padding_mask


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
    Injects positional information into the input embeddings.

    This implementation pre-computes the positional encoding matrix in __init__
    using the sine and cosine functions described in "Attention Is All You Need".
    It uses register_buffer to store the encoding matrix efficiently.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float,
        max_len: int = 5000,  # Maximum possible sequence length
        device: torch.device = device,
    ):
        """
        Initializes the PositionalEncoding layer.

        Args:
            d_model (int): The dimensionality of the embeddings (must match input).
            dropout (float): The dropout rate to apply after adding positional encoding.
            max_len (int): The maximum sequence length that this model might ever encounter.
                           The positional encoding matrix will be pre-computed up to this length.
            device (torch.device): The device to place tensors on.
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len
        self.device = device

        # --- Pre-compute the positional encoding matrix ---
        # Shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model, device=self.device)

        # Position indices: (max_len, 1)
        position = torch.arange(
            0, max_len, dtype=torch.float, device=self.device
        ).unsqueeze(1)

        # Calculate the division term: (d_model / 2)
        # Formula: 1 / (10000^(2i / d_model)) -> log space -> exp(- (2i / d_model) * log(10000))
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float, device=self.device)
            * (-math.log(10000.0) / d_model)
        )

        # Calculate sine for even indices and cosine for odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply to even columns
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply to odd columns

        # Add a batch dimension for broadcasting: (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Register 'pe' as a buffer. Buffers are model state like parameters,
        # but are not updated by the optimizer. They are saved with the model
        # and moved to the correct device automatically with .to(device).
        self.register_buffer("pe", pe)
        print(
            f"PositionalEncoding: Pre-computed 'pe' buffer with shape {self.pe.shape}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input embeddings.

        Args:
            x (torch.Tensor): The input embeddings tensor of shape
                              (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The embeddings tensor with positional information added,
                          after applying dropout. Shape remains
                          (batch_size, seq_len, d_model).
        """
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape

        # Ensure seq_len does not exceed max_len
        if seq_len > self.max_len:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds the maximum "
                f"sequence length ({self.max_len}) this PositionalEncoding "
                f"was initialized with."
            )

        # Add the pre-computed positional encoding.
        # We take only the first 'seq_len' positions from the pre-computed matrix.
        # self.pe shape: (1, max_len, d_model)
        # Sliced pe shape: (1, seq_len, d_model)
        # This will broadcast correctly across the batch dimension of x.
        x = x + self.pe[:, :seq_len, :]  # Add positional encoding

        # Apply dropout
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        # mask: torch.Tensor | None, # mask should be passed in the forward method
        dropout: float,
        device: torch.device = device,
    ) -> None:
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
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def scaled_dot_product_attention(
        query,
        key,
        value,
        dropout: Optional[nn.Dropout],
        mask=None,
        device: torch.device = "cpu",
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # Return weights too
        batch_size, num_heads, seq_len_q, d_k = query.shape
        seq_len_k = key.shape[2]

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # attn_scores shape: (batch_size, num_heads, seq_len_q, seq_len_k)

        if mask is not None:
            mask = mask.to(device)

            # The mask needs to be broadcastable to attn_scores shape.
            # Common valid shapes include:
            # (B, 1, 1, Sk) - Padding mask for Encoder/Cross-Attention
            # (B, 1, Sq, Sk) - Combined mask for Decoder Self-Attention
            # (1, 1, Sq, Sk) - Look-ahead mask only
            # I mainly need the last two dimensions to match or broadcast correctly.
            if mask.dim() != 4:
                raise ValueError(f"Mask dimension must be 4, got {mask.dim()}")
            if mask.shape[-1] != seq_len_k:
                raise ValueError(
                    f"Mask key sequence length {mask.shape[-1]} doesn't match scores key length {seq_len_k}"
                )
            if (
                mask.shape[-2] != seq_len_q and mask.shape[-2] != 1
            ):  # Allow query dim to be 1 for broadcasting
                raise ValueError(
                    f"Mask query sequence length {mask.shape[-2]} is incompatible with scores query length {seq_len_q}"
                )

            # Mask should be True where attention is inhibited (padding or future positions)
            attn_scores = attn_scores.masked_fill(
                mask, float("-1e9")
            )  # Use large negative number

        attn_weights = torch.softmax(attn_scores, dim=-1)
        if dropout is not None:
            attn_weights = dropout(attn_weights)
        output = torch.matmul(attn_weights, value)
        return output, attn_weights  # Return weights

    def forward(
        self,
        q_encodings: torch.Tensor,
        k_encodings: torch.Tensor,
        v_encodings: torch.Tensor,
        mask=None,
        return_weights: bool = False,
    ) -> Tuple[
        torch.Tensor, Optional[torch.Tensor]
    ]:  # return attention weights if requested

        q_batch_size, q_seq_len, _ = q_encodings.shape
        k_batch_size, k_seq_len, _ = k_encodings.shape
        v_batch_size, v_seq_len, _ = v_encodings.shape
        # these 3 ↑ are pretty much the same, but I'm writing them like this for my own clarity

        # (batch, seq_length, d_model) -> (batch, seq_length, d_model)
        q = self.W_q(q_encodings)

        # (batch, seq_length, d_model) -> (batch, seq_length, d_model)
        k = self.W_k(k_encodings)

        # (batch, seq_length, d_model) -> (batch, seq_length, d_model)
        v = self.W_v(v_encodings)

        query = q.view(q_batch_size, q_seq_len, self.num_heads, self.d_k).transpose(
            1, 2
        )
        # ========================== ↑ Query Tensor Reshape Logic ↑ ==========================
        # (batch, seq_length, d_model) {view} -> (batch, seq_length, num_heads, d_k) {transpose} -> (batch, num_heads, seq_length, d_k)

        key = k.view(k_batch_size, k_seq_len, self.num_heads, self.d_k).transpose(1, 2)
        value = v.view(v_batch_size, v_seq_len, self.num_heads, self.d_k).transpose(
            1, 2
        )

        # shape of output => (batch, num_heads, seq_length, d_k)
        output, attn_weights = MultiHeadAttention.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            dropout=self.dropout,
            mask=mask,
            device=self.device,
        )

        H = torch.transpose(output, 1, 2).contiguous()
        H = H.view(H.shape[0], -1, H.shape[-1] * H.shape[-2])
        # ========================== ↑ H (concatenated head) Tensor Logic ↑ ==========================
        # (batch, num_heads, seq_length, d_k) {transpose} -> (batch, seq_length, num_heads, d_k) {contiguous} -> (batch, seq_length, num_heads * d_k)

        final_output = self.W_o(H)
        if return_weights:
            return final_output, attn_weights  # Return weights if requested
        else:
            return final_output, None  # Return None for weights otherwise


class ResidualConnection(nn.Module):
    """
    Residual connection for the encoder/decoder block.

    Arguments:

        Inputs:

            1. x: tensor, the input to the residual connection
            2. sublayer: the sublayer of the encoder/decoder block, i.e, either the multi-head attention mechanism or the feed-forward network.

        Outputs:
            1. tensor, the output of the residual connection

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

    def __init__(self, droput: float, d_model: int, device: torch.device = device):
        super().__init__()
        self.device = device
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(droput)

    def forward(self, x: torch.Tensor, sublayer: nn.Module):
        """
        Args:
            x: tensor, the input to the sublayer
            sublayer: the sublayer of the encoder/decoder block, i.e, either the
            multi-head attention mechanism or the feed-forward network.

        Returns:
            tensor, the output of the sublayer

        Note:
            What I am doing here is the actual implementation based on the original paper.
            However, it seems the more common implementation is to do the following:
            ```
            normalized = self.norm(x)
            sublayer_output = sublayer(normalized)
            dropped = self.dropout(sublayer_output)
            return x + dropped
            ```
        This is probably what I will do in the future, but for now, I am sticking to the original implementation.
        """
        sublayer_instance = sublayer(x)
        dropped_output = self.dropout(sublayer_instance)
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
        normalized_x = (x - mean_x) / torch.sqrt(var_x + self.epsilon)
        return self.gamma * normalized_x + self.beta

    # NOTE: I could just use nn.LayerNorm, but I implemented it myself to understand it better.
    # nn.LayerNorm is probably better if I'm using it for more serious projects.


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


class EncoderBlock(nn.Module):
    """
    Transformer encoder block (individual unit in the encoder stack).

    This block processes already embedded inputs and applies:
    1. Multi-head self-attention with residual connection and layer normalization
    2. Feed-forward network with residual connection and layer normalization

    It does NOT handle tokenization or embedding - that's the Encoder's job.
    """

    def __init__(
        self,
        config: dict,
        self_attn_block: MultiHeadAttention,
        feed_forward_block: FeedForward,
        device: torch.device = device,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device

        self.self_attn_block = self_attn_block
        self.feed_forward_block = feed_forward_block
        """
        Here, `self_attn_block` and `feed_forward_block` are arguments to the
        `__init__` method. This means that when I create an `Encoder` object
        (e.g. `encoder = Encoder(config, self_attn_block, feed_forward_block, device)`),
        I am required to provide pre-existing instances of `MultiHeadAttention`
        and `FeedForward`. The `Encoder` class itself is not responsible for
        creating these objects. It depends on them being created and passed
        in from the outside. This is "dependency injection" - I am "injecting"
        the dependencies (MultiHeadAttention, FeedForward) into the Encoder.
        """

        self.residual_connections = nn.ModuleList(
            [
                ResidualConnection(config["dropout"], config["d_model"], device=device)
                for _ in range(2)
            ]
        )

    def forward(
        self, x, mask, return_attn_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the encoder block.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, seq_len, d_model].
            mask (torch.Tensor): Padding mask of shape [batch, 1, 1, seq_len].
            return_attn_weights (bool): If True, return attention weights from MHA.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Output tensor of the same shape as input.
                - Attention weights tensor if return_attn_weights is True, else None.
        """
        mha_sublayer = lambda x_residual: self.self_attn_block(
            x_residual,
            x_residual,
            x_residual,
            mask,
            return_weights=return_attn_weights,  # Passing flag
        )

        # Need to handle the tuple output (output, weights) from mha_sublayer
        attn_block_input = x
        attn_raw_output, attn_weights = mha_sublayer(
            attn_block_input
        )  # Getting both outputs

        # first connection (mha + dropout + residual)
        x = self.residual_connections[0].norm(
            attn_block_input + self.residual_connections[0].dropout(attn_raw_output)
        )

        # Storing the output of the first residual connection for the second one
        ff_block_input = x

        # second connection (ff + dropout + residual)
        # FeedForward doesn't need the mask or return weights
        ff_sublayer = self.feed_forward_block
        ff_raw_output = ff_sublayer(ff_block_input)
        x = self.residual_connections[1].norm(
            ff_block_input + self.residual_connections[1].dropout(ff_raw_output)
        )

        return x, attn_weights  # Returning final output and weights (or None)


class Encoder(nn.Module):
    """
    The main Transformer Encoder stack.

    Handles input embedding, positional encoding, and passes the data
    through multiple EncoderBlock layers.
    """

    def __init__(self, config: dict, n_layers: int = 6, device: torch.device = device):
        """
        Initializes the Encoder stack.

        Args:
            config (dict): Configuration dictionary containing parameters like:
                           'd_model', 'tokenizer_name', 'dropout', 'max_len' (for PE),
                           'num_heads', 'd_ff'.
            n_layers (int): Number of EncoderBlocks to stack. Defaults to 6.
            device (torch.device): The device to place tensors on.
        """
        super().__init__()
        self.config = config
        self.device = device
        self.n_layers = n_layers

        # Input processing layers
        self.input_embedding = InputEmbedding(
            d_model=config["d_model"],
            tokenizer_name=config.get(
                "tokenizer_name", DEFAULT_TOKENIZER_NAME
            ),  # Getting from config or using default
            device=device,
            max_length=config.get("max_length", None),  # Allowing fixed length via config
        )
        self.positional_encoding = PositionalEncoding(
            d_model=config["d_model"],
            dropout=config["dropout"],
            max_len=config.get("max_len_pe", 5000),  # Max length for PE matrix
            device=device,
        )

        # Creating a stack of n_layers encoder blocks
        self.layers = nn.ModuleList(
            [self._create_encoder_block() for _ in range(n_layers)]
        )

        # self._seq_length = None

    def _create_encoder_block(self) -> EncoderBlock:
        """Helper method to create a single EncoderBlock."""
        self_attn = MultiHeadAttention(
            num_heads=self.config["num_heads"],
            d_model=self.config["d_model"],
            # mask=None, # Mask is passed in forward so don't need to pass here
            dropout=self.config["dropout"],
            device=self.device,
        )
        feed_forward = FeedForward(
            d_model=self.config["d_model"],
            d_ff=self.config.get("d_ff", 2048),  # Default d_ff
            dropout=self.config["dropout"],
            device=self.device,
        )
        return EncoderBlock(
            config=self.config,  # Passing config down if needed by block/sublayers
            self_attn_block=self_attn,
            feed_forward_block=feed_forward,
            device=self.device,
        )

    def forward(
        self, input_texts: List[str], return_last_layer_attn_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the entire Encoder stack.

        Args:
            input_texts (List[str]): A batch of raw input strings.
            return_last_layer_attn_weights (bool): If True, return attention weights
                                             from the final EncoderBlock.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - The encoded output tensor of shape (batch_size, seq_len, d_model).
                - Attention weights from the last layer if requested, else None.
        """
        # 1. Getting embeddings and the padding mask
        # embeddings: (batch, seq_len, d_model)
        # padding_mask: (batch, 1, 1, seq_len), True for padding
        embeddings, padding_mask = self.input_embedding(input_texts)
        print(
            f"Encoder.forward: Embeddings shape: {embeddings.shape}, Padding Mask shape: {padding_mask.shape}"
        )

        # 2. Adding positional encoding
        x = self.positional_encoding(embeddings)
        print(f"Encoder.forward: After Positional Encoding shape: {x.shape}")

        last_layer_attn_weights = None  # Initialize to None

        # 3. Passing through each encoder block layer
        for i, layer in enumerate(self.layers):
            # Requesting weights only from the last layer if needed
            request_weights_from_layer = return_last_layer_attn_weights and (
                i == self.n_layers - 1
            )
            x, attn_weights = layer(
                x, padding_mask, return_attn_weights=request_weights_from_layer
            )
            print(f"Encoder.forward: After Layer {i+1} shape: {x.shape}")
            if request_weights_from_layer:
                last_layer_attn_weights = attn_weights

        print(f"Encoder.forward: Final Output shape: {x.shape}")
        return x, last_layer_attn_weights

    # Exposing tokenizer for testing convenience
    @property
    def tokenizer(self):
        return self.input_embedding.tokenizer

    # Exposing padding ID for testing convenience
    @property
    def padding_token_id(self):
        return self.input_embedding.padding_token_id


class DecoderBlock(nn.Module):
    """
    Single block for the Transformer Decoder stack.

    Applies Masked Self-Attention, Cross-Attention, and Feed-Forward network,
    each followed by Add & Norm (Post-LN).
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        device: torch.device,
    ):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(num_heads, d_model, dropout, device)
        self.cross_attn = MultiHeadAttention(num_heads, d_model, dropout, device)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, device)

        self.norm1 = LayerNorm(d_model).to(device) # Ensuring LayerNorm is also moved to device if custom
        self.norm2 = LayerNorm(d_model).to(device)
        self.norm3 = LayerNorm(d_model).to(device)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                 # Target sequence embeddings (B, TgtSeqLen, D)
        encoder_output: torch.Tensor,    # Encoder output (B, SrcSeqLen, D)
        target_mask: torch.Tensor,       # Combined causal & padding mask for target (B, 1, TgtSeqLen, TgtSeqLen) or broadcastable
        encoder_padding_mask: torch.Tensor,# Padding mask for encoder output (B, 1, 1, SrcSeqLen)
        return_attn_weights: bool = False # ADDED parameter
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]: # MODIFIED return type
        # Returning output, self_attn_weights, cross_attn_weights

        # 1. Masked Self-Attention (Query, Key, Value are all x)
        # Passing the return_weights flag down
        self_attn_output, self_attn_weights = self.masked_self_attn(
            x, x, x, target_mask, return_weights=return_attn_weights
        )
        x = self.norm1(x + self.dropout(self_attn_output))  # Add & Norm

        # 2. Cross-Attention (Query is x, Key/Value are encoder_output)
        # Passing the return_weights flag down
        cross_attn_output, cross_attn_weights = self.cross_attn(
            x, encoder_output, encoder_output, encoder_padding_mask, return_weights=return_attn_weights
        )
        x = self.norm2(x + self.dropout(cross_attn_output))  # Add & Norm

        # 3. Feed Forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))  # Add & Norm

        # Returning weights only if requested
        sa_w = self_attn_weights if return_attn_weights else None
        ca_w = cross_attn_weights if return_attn_weights else None
        return x, sa_w, ca_w


class Decoder(nn.Module):
    """
    The Transformer Decoder stack.
    Handles target embedding, positional encoding, and passes data through
    multiple DecoderBlock layers.
    """

    def __init__(
        self,
        target_vocab_size: int,
        d_model: int,
        n_layers: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        max_len_pe: int,
        device: torch.device,
    ):
        super().__init__()
        self.device = device
        self.n_layers = n_layers
        self.target_embedding = nn.Embedding(target_vocab_size, d_model, device=device)
        self.positional_encoding = PositionalEncoding(
            d_model, dropout, max_len_pe, device
        )

        self.layers = nn.ModuleList(
            [
                DecoderBlock(d_model, num_heads, d_ff, dropout, device)
                for _ in range(n_layers)
            ]
        )
        print(f"Decoder: Initialized with {n_layers} layers.")

    def _create_look_ahead_mask(self, size: int) -> torch.Tensor:
        """Creates a look-ahead mask of shape (1, 1, size, size)."""
        mask = torch.ones(size, size, device=self.device, dtype=torch.bool)
        mask = torch.triu(mask, diagonal=1)  # Upper triangle (True where j > i)
        return mask.unsqueeze(0).unsqueeze(0)  # Adding batch and head dims

    def forward(
        self,
        target_token_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
        target_padding_mask: Optional[torch.Tensor] = None,
        return_last_layer_attn_weights: bool = False,  # Flag to get weights from last layer
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Returning output, last_self_attn_weights, last_cross_attn_weights

        batch_size, tgt_seq_len = target_token_ids.shape
        x = self.target_embedding(target_token_ids) * math.sqrt(
            self.target_embedding.embedding_dim
        )
        x = self.positional_encoding(x)

        look_ahead_mask = self._create_look_ahead_mask(tgt_seq_len)
        combined_target_mask = (
            torch.logical_or(look_ahead_mask, target_padding_mask)
            if target_padding_mask is not None
            else look_ahead_mask
        )

        last_self_attn_weights = None
        last_cross_attn_weights = None
        for i, layer in enumerate(self.layers):
            request_weights = return_last_layer_attn_weights and (
                i == self.n_layers - 1
            )
            x, sa_w, ca_w = layer(  # Getting all 3 return values
                x=x,
                encoder_output=encoder_output,
                target_mask=combined_target_mask,
                encoder_padding_mask=encoder_padding_mask,
                return_attn_weights=request_weights,  # Passing flag to layer
            )
            if request_weights:
                last_self_attn_weights = sa_w
                last_cross_attn_weights = ca_w

        return x, last_self_attn_weights, last_cross_attn_weights


class ProjectionLayer(nn.Module):
    """Projects the decoder output to vocabulary logits."""

    def __init__(self, d_model: int, vocab_size: int, device: torch.device):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (Batch, SeqLen, Dim) -> (Batch, SeqLen, VocabSize)
        # Applying log_softmax for NLLLoss or keeping as logits for CrossEntropyLoss
        # return F.log_softmax(self.proj(x), dim=-1) # Example for NLLLoss
        return self.proj(x)  # Returning raw logits for CrossEntropyLoss


class Transformer(nn.Module):
    """
    A standard Encoder-Decoder Transformer model.
    """

    def __init__(
        self, encoder: nn.Module, decoder: nn.Module, projection_layer: nn.Module
    ):
        """
        Initializes the full Transformer model.

        Args:
            encoder (nn.Module): An initialized Encoder instance.
            decoder (nn.Module): An initialized Decoder instance.
            projection_layer (nn.Module): An initialized ProjectionLayer instance.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.projection_layer = projection_layer

    def forward(
        self,
        source_texts: List[str],
        target_token_ids: torch.Tensor,
        target_padding_id: int = 0,
        return_last_dec_attn_weights: bool = False,  # Flag for decoder weights
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Returning logits, last_self_attn_weights, last_cross_attn_weights

        # 1. Encoding
        # *** ENSURING MY ENCODER RETURNS (output, mask) ***
        encoder_output, encoder_padding_mask = self.encoder(source_texts)

        # 2. Preparing target padding mask
        target_padding_mask = (
            (target_token_ids == target_padding_id)
            .unsqueeze(1)
            .unsqueeze(2)
            .to(target_token_ids.device)
        )

        # 3. Decoding
        decoder_output, last_self_attn_weights, last_cross_attn_weights = self.decoder(
            target_token_ids=target_token_ids,
            encoder_output=encoder_output,
            encoder_padding_mask=encoder_padding_mask,
            target_padding_mask=target_padding_mask,
            # Passing the flag down to the decoder
            return_last_layer_attn_weights=return_last_dec_attn_weights,
        )

        # 4. Projecting to Logits
        logits = self.projection_layer(decoder_output)

        # Returning logits and optionally the weights from the last decoder layer
        return logits, last_self_attn_weights, last_cross_attn_weights