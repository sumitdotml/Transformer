import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from transformers import AutoTokenizer

# Set default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LayerNormalization(nn.Module):
    """
    Layer normalization module (with learnable parameters)
    """
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))  # Scale parameter
        self.bias = nn.Parameter(torch.zeros(features))  # Shift parameter

    def forward(self, x):
        # x: (batch, seq_len, features)
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)    # (batch, seq_len, 1)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    """
    Position-wise feed-forward network
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(F.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-head attention block
    """
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model  # Embedding dimension
        self.h = h              # Number of heads
        assert d_model % h == 0, "d_model must be divisible by h"
        
        self.d_k = d_model // h  # Dimension of each head
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: Optional[nn.Dropout]):
        d_k = query.shape[-1]
        
        # (batch, h, seq_len, d_k) @ (batch, h, d_k, seq_len) -> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            # Apply mask (0 for positions to mask, 1 for positions to keep)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        if dropout is not None:
            attention_weights = dropout(attention_weights)
            
        # (batch, h, seq_len, seq_len) @ (batch, h, seq_len, d_k) -> (batch, h, seq_len, d_k)
        return (attention_weights @ value), attention_weights

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        
        # Linear projections
        query = self.w_q(q)  # (batch, seq_len, d_model)
        key = self.w_k(k)    # (batch, seq_len, d_model)
        value = self.w_v(v)  # (batch, seq_len, d_model)
        
        # Reshape for multi-head attention
        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        query = query.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        
        # Apply attention
        x, self.attention_weights = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )
        
        # Combine heads and project
        # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        return self.w_o(x)

class ResidualConnection(nn.Module):
    """
    Residual connection with layer normalization and dropout
    Uses pre-norm architecture: Norm -> Sublayer -> Dropout -> Add
    """
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.norm = LayerNormalization(features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        # Apply normalization first, then sublayer, then dropout, then add residual
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    """
    Single encoder block with self-attention and feed-forward network
    """
    def __init__(self, d_model: int, self_attention: MultiHeadAttentionBlock, 
                 feed_forward: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([
            ResidualConnection(d_model, dropout) for _ in range(2)
        ])
        
    def forward(self, x, src_mask=None):
        # Self-attention with residual connection
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x, src_mask))
        
        # Feed-forward with residual connection
        x = self.residual_connections[1](x, self.feed_forward)
        
        return x

class Encoder(nn.Module):
    """
    Full encoder stack with embedding and positional encoding
    """
    def __init__(self, d_model: int, vocab_size: int, seq_len: int,
                 n_layers: int, h: int, d_ff: int, dropout: float, 
                 tokenizer_name: str = "bert-base-uncased"):
        super().__init__()
        
        # Input embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, seq_len, dropout)
        
        # Encoder blocks
        self.blocks = nn.ModuleList([
            self.build_encoder_block(d_model, h, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.norm = LayerNormalization(d_model)
        
        # Tokenizer for processing raw text
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.pad_token_id = self.tokenizer.pad_token_id
        
    @staticmethod
    def build_encoder_block(d_model, h, d_ff, dropout):
        return EncoderBlock(
            d_model=d_model,
            self_attention=MultiHeadAttentionBlock(d_model, h, dropout),
            feed_forward=FeedForwardBlock(d_model, d_ff, dropout),
            dropout=dropout
        )
        
    def forward(self, x, mask=None):
        """
        Forward pass for the encoder
        
        Args:
            x: Either token ids (batch, seq_len) or List[str] of input texts
            mask: Optional mask for padding (batch, 1, 1, seq_len)
            
        Returns:
            Encoder output and mask
        """
        # Process raw text input if needed
        if isinstance(x, list) and isinstance(x[0], str):
            return self.forward_text(x)
            
        # Token embeddings 
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply encoder blocks
        for block in self.blocks:
            x = block(x, mask)
            
        # Apply final layer norm
        return self.norm(x), mask
    
    def forward_text(self, texts: List[str]):
        """Process raw text input"""
        # Tokenize input texts
        encoding = self.tokenizer(
            texts,
            padding="longest",
            truncation=True, 
            return_tensors="pt",
            return_attention_mask=True
        ).to(device)
        
        # Extract token IDs and attention mask
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        
        # Create padding mask for attention mechanism (1 for positions to attend, 0 for padding)
        padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Embed tokens
        x = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply encoder blocks
        for block in self.blocks:
            x = block(x, padding_mask)
            
        # Apply final layer norm
        return self.norm(x), padding_mask

class DecoderBlock(nn.Module):
    """
    Single decoder block with masked self-attention, cross-attention, and feed-forward network
    """
    def __init__(self, d_model: int, self_attention: MultiHeadAttentionBlock,
                 cross_attention: MultiHeadAttentionBlock, feed_forward: FeedForwardBlock,
                 dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([
            ResidualConnection(d_model, dropout) for _ in range(3)
        ])
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention with residual connection
        x = self.residual_connections[0](
            x, lambda x: self.self_attention(x, x, x, tgt_mask)
        )
        
        # Cross-attention with residual connection
        x = self.residual_connections[1](
            x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask)
        )
        
        # Feed-forward with residual connection
        x = self.residual_connections[2](x, self.feed_forward)
        
        return x

class Decoder(nn.Module):
    """
    Full decoder stack with embedding and positional encoding
    """
    def __init__(self, d_model: int, vocab_size: int, seq_len: int,
                 n_layers: int, h: int, d_ff: int, dropout: float):
        super().__init__()
        
        # Input embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, seq_len, dropout)
        
        # Decoder blocks
        self.blocks = nn.ModuleList([
            self.build_decoder_block(d_model, h, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.norm = LayerNormalization(d_model)
        
    @staticmethod
    def build_decoder_block(d_model, h, d_ff, dropout):
        return DecoderBlock(
            d_model=d_model,
            self_attention=MultiHeadAttentionBlock(d_model, h, dropout),
            cross_attention=MultiHeadAttentionBlock(d_model, h, dropout),
            feed_forward=FeedForwardBlock(d_model, d_ff, dropout),
            dropout=dropout
        )
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Forward pass for the decoder
        
        Args:
            x: Target token ids (batch, seq_len)
            encoder_output: Output from encoder (batch, seq_len, d_model)
            src_mask: Mask for padding in the encoder output
            tgt_mask: Combined padding and causal mask for the target
            
        Returns:
            Decoder output
        """
        # Token embeddings
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply decoder blocks
        for block in self.blocks:
            x = block(x, encoder_output, src_mask, tgt_mask)
            
        # Apply final layer norm
        return self.norm(x)
    
    def create_target_mask(self, tgt, padding_idx=0):
        """
        Create both padding and look-ahead masks for the target
        
        Args:
            tgt: Target token ids (batch, seq_len)
            padding_idx: Token ID used for padding
            
        Returns:
            Combined mask for target (batch, 1, seq_len, seq_len)
        """
        # Padding mask
        padding_mask = (tgt == padding_idx).unsqueeze(1).unsqueeze(2)
        
        # Look-ahead mask
        seq_len = tgt.size(1)
        look_ahead_mask = (1 - torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.uint8, device=tgt.device),
            diagonal=1
        )).unsqueeze(0).unsqueeze(0).bool()
        
        # Combine masks
        return padding_mask | ~look_ahead_mask

class PositionalEncoding(nn.Module):
    """
    Positional encoding using sine and cosine functions
    """
    def __init__(self, d_model: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a model parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input embeddings
        
        Args:
            x: Input embeddings (batch, seq_len, d_model)
            
        Returns:
            Embeddings with positional information
        """
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)

class ProjectionLayer(nn.Module):
    """
    Final projection layer that converts decoder output to vocabulary logits
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        """
        Project decoder output to vocabulary logits
        
        Args:
            x: Decoder output (batch, seq_len, d_model)
            
        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        logits = self.proj(x)
        return F.log_softmax(logits, dim=-1)

class Transformer(nn.Module):
    """
    Complete Transformer model
    """
    def __init__(self, encoder: Encoder, decoder: Decoder, projection: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.projection = projection
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass through the entire Transformer
        
        Args:
            src: Source input (token IDs or raw text)
            tgt: Target input (token IDs)
            src_mask: Source padding mask
            tgt_mask: Target combined mask
            
        Returns:
            Output logits
        """
        # Create target mask if not provided
        if tgt_mask is None:
            tgt_mask = self.decoder.create_target_mask(tgt)
            
        # Encode the source
        encoder_output, src_mask = self.encoder(src, src_mask)
        
        # Decode with target and encoder output
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        
        # Project to vocabulary
        return self.projection(decoder_output)
    
    @torch.no_grad()
    def generate(self, src, max_len=50, start_symbol=101, end_symbol=102):
        """
        Generate a sequence using the trained model
        
        Args:
            src: Source input (token IDs or raw text)
            max_len: Maximum length of the generated sequence
            start_symbol: Start token ID
            end_symbol: End token ID
            
        Returns:
            Generated sequence
        """
        # Encode the source
        encoder_output, src_mask = self.encoder(src, None)
        
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # Initialize target with start symbol
        ys = torch.ones(batch_size, 1).fill_(start_symbol).long().to(device)
        
        # Generate tokens one by one
        for i in range(max_len-1):
            # Create mask for target
            tgt_mask = self.decoder.create_target_mask(ys)
            
            # Decode current sequence
            decoder_output = self.decoder(ys, encoder_output, src_mask, tgt_mask)
            
            # Get next token (last position only)
            logits = self.projection(decoder_output)
            next_word = logits[:, -1].argmax(dim=-1, keepdim=True)
            
            # Append to target sequence
            ys = torch.cat([ys, next_word], dim=1)
            
            # Stop if all sequences have end token
            if (next_word == end_symbol).all():
                break
                
        return ys

def build_transformer(
    src_vocab_size: int, 
    tgt_vocab_size: int, 
    src_seq_len: int, 
    tgt_seq_len: int, 
    d_model: int = 512, 
    n_layers: int = 6, 
    n_heads: int = 8,
    d_ff: int = 2048, 
    dropout: float = 0.1,
    tokenizer_name: str = "bert-base-uncased"
) -> Transformer:
    """
    Build a complete Transformer model
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        src_seq_len: Source sequence length
        tgt_seq_len: Target sequence length
        d_model: Model dimension
        n_layers: Number of encoder/decoder layers
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        tokenizer_name: Name of tokenizer to use
        
    Returns:
        Transformer model
    """
    # Build encoder
    encoder = Encoder(
        d_model=d_model,
        vocab_size=src_vocab_size,
        seq_len=src_seq_len,
        n_layers=n_layers,
        h=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        tokenizer_name=tokenizer_name
    )
    
    # Build decoder
    decoder = Decoder(
        d_model=d_model,
        vocab_size=tgt_vocab_size,
        seq_len=tgt_seq_len,
        n_layers=n_layers,
        h=n_heads,
        d_ff=d_ff,
        dropout=dropout
    )
    
    # Build projection layer
    projection = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Build transformer
    transformer = Transformer(encoder, decoder, projection)
    
    # Initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return transformer

def build_transformer_from_config(config):
    """
    Build a transformer model from a configuration dictionary
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Transformer model
    """
    # Get tokenizer to determine vocab size
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    vocab_size = len(tokenizer)
    
    return build_transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        src_seq_len=config["max_seq_len"],
        tgt_seq_len=config["max_seq_len"],
        d_model=config["d_model"],
        n_layers=config["num_layers"],
        n_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        tokenizer_name=config["tokenizer_name"]
    )

if __name__ == "__main__":
    # Import configuration
    from config import get_config
    
    # Get default config
    config = get_config()
    
    # Build model
    model = build_transformer_from_config(config)
    
    # Test with some example data
    src_texts = ["Hello, world!", "This is a test."]
    tgt_tokens = torch.ones(2, 5).long() * 101  # Example target tokens
    
    # Run forward pass
    output = model(src_texts, tgt_tokens)
    
    # Generate sequence
    generated = model.generate(src_texts)
    
    # Print shapes
    print(f"Output shape: {output.shape}")
    print(f"Generated shape: {generated.shape}")
    print("Test successful!") 