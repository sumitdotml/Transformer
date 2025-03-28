import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from transformers import AutoTokenizer

from config import get_device_config

device = get_device_config()

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
            # Applying mask (0 for positions to mask, 1 for positions to keep)
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
        # Applying softmax to get attention weights
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
        
        # Reshaping for multi-head attention
        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        query = query.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        
        x, self.attention_weights = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )
        
        # Combining heads and projecting
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
        # Applying normalization first, then sublayer, then dropout, then adding residual
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
                 tokenizer_name: str = "bert-base-uncased",
                 max_len_pe: int = None):
        super().__init__()
        
        # Determining appropriate device
        self.device = device  # Using the centralized device configuration
        
        # Using provided max_len_pe or defaulting to seq_len
        max_len_pe = max_len_pe or seq_len
        
        # Input embeddings - wrapping in try-except for robust device handling
        try:
            self.embedding = nn.Embedding(vocab_size, d_model, device=self.device)
        except RuntimeError as e:
            print(f"Warning: Could not create embedding directly on {self.device}: {e}")
            print("Creating embedding on CPU and moving to target device")
            self.embedding = nn.Embedding(vocab_size, d_model).to(self.device)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len_pe, dropout)
        
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
        # Processing raw text input if needed
        if isinstance(x, list) and isinstance(x[0], str):
            return self.forward_text(x)
            
        # Ensuring input is on the correct device
        if x.device != self.device:
            x = x.to(self.device)
            
        # Token embeddings 
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        
        # Adding positional encoding
        x = self.positional_encoding(x)
        
        # Applying encoder blocks
        for block in self.blocks:
            x = block(x, mask)
            
        # Applying final layer norm
        return self.norm(x), mask
    
    def forward_text(self, texts: List[str]):
        """Process raw text input"""
        # Tokenizing input texts
        encoding = self.tokenizer(
            texts,
            padding="longest",
            truncation=True, 
            return_tensors="pt"
        )
        
        # Ensuring encoded tensors are on the correct device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        # Extracting token IDs and attention mask
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        
        # Creating padding mask for attention mechanism (1 for positions to attend, 0 for padding)
        padding_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Embedding tokens
        x = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        
        # Adding positional encoding
        x = self.positional_encoding(x)
        
        # Applying encoder blocks
        for block in self.blocks:
            x = block(x, padding_mask)
            
        # Applying final layer norm
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
                 n_layers: int, h: int, d_ff: int, dropout: float,
                 max_len_pe: int = None):
        super().__init__()
        
        self.device = device 
        
        # Using provided max_len_pe or defaulting to seq_len
        max_len_pe = max_len_pe or seq_len
        
        # Input embeddings - wrapping in try-except for robust device handling
        try:
            # Trying to create embeddings directly on target device
            self.embedding = nn.Embedding(vocab_size, d_model, device=self.device)
        except RuntimeError as e:
            print(f"Warning: Could not create decoder embedding directly on {self.device}: {e}")
            print("Creating embedding on CPU and moving to target device")
            self.embedding = nn.Embedding(vocab_size, d_model).to(self.device)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len_pe, dropout)
        
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
        # Ensuring inputs are on the correct device
        if x.device != self.device:
            x = x.to(self.device)
        if encoder_output.device != self.device:
            encoder_output = encoder_output.to(self.device)
        if src_mask is not None and src_mask.device != self.device:
            src_mask = src_mask.to(self.device)
        if tgt_mask is not None and tgt_mask.device != self.device:
            tgt_mask = tgt_mask.to(self.device)
            
        # Token embeddings
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        
        # Adding positional encoding
        x = self.positional_encoding(x)
        
        # Applying decoder blocks
        for block in self.blocks:
            x = block(x, encoder_output, src_mask, tgt_mask)
            
        # Applying final layer norm
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
        # Ensuring input is on the correct device
        if tgt.device != self.device:
            tgt = tgt.to(self.device)
            
        padding_mask = (tgt == padding_idx).unsqueeze(1).unsqueeze(2)
        
        # Look-ahead mask - creating on the same device as the input
        seq_len = tgt.size(1)
        look_ahead_mask = (1 - torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.uint8, device=tgt.device),
            diagonal=1
        )).unsqueeze(0).unsqueeze(0).bool()
        
        # Combining masks
        return padding_mask | ~look_ahead_mask

class PositionalEncoding(nn.Module):
    """
    Positional encoding using sine and cosine functions
    """
    def __init__(self, d_model: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.device = device 
        
        try:
            # Trying to create positional encoding on the target device
            # Creating positional encoding matrix
            pe = torch.zeros(max_seq_len, d_model, device=self.device)
            position = torch.arange(0, max_seq_len, dtype=torch.float, device=self.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=self.device) * (-math.log(10000.0) / d_model))
        except RuntimeError as e:
            # Falling back to CPU if device creation fails
            print(f"Warning: Could not create positional encoding on {self.device}: {e}")
            print("Creating on CPU and moving to target device")
            pe = torch.zeros(max_seq_len, d_model)
            position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        
        # Applying sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Applying cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Adding batch dimension
        pe = pe.unsqueeze(0)
        
        # Registering as buffer (not a model parameter)
        self.register_buffer('pe', pe.to(self.device))
        
    def forward(self, x):
        """
        Add positional encoding to input embeddings
        
        Args:
            x: Input embeddings (batch, seq_len, d_model)
            
        Returns:
            Embeddings with positional information
        """
        # Ensuring PE buffer is on the same device as the input
        if self.pe.device != x.device:
            print(f"Moving positional encoding from {self.pe.device} to {x.device}")
            self.pe = self.pe.to(x.device)
            
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)

class ProjectionLayer(nn.Module):
    """
    Final projection layer that converts decoder output to vocabulary logits
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.device = device 
        
        # Trying to create projection layer on the target device
        try:
            self.proj = nn.Linear(d_model, vocab_size, device=self.device)
        except RuntimeError as e:
            print(f"Warning: Could not create projection layer directly on {self.device}: {e}")
            print("Creating projection layer on CPU and moving to target device")
            self.proj = nn.Linear(d_model, vocab_size).to(self.device)
        
    def forward(self, x):
        """
        Project decoder output to vocabulary logits
        
        Args:
            x: Decoder output (batch, seq_len, d_model)
            
        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        # Ensuring input is on the correct device
        if x.device != self.device:
            x = x.to(self.device)
            
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
        # Ensuring consistent device handling
        self.device = device
        
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
        if isinstance(tgt, torch.Tensor) and tgt.device != self.device:
            tgt = tgt.to(self.device)
            
        # Creating target mask if not provided
        if tgt_mask is None:
            tgt_mask = self.decoder.create_target_mask(tgt)
            
        # Encoding the source
        encoder_output, src_mask = self.encoder(src, src_mask)
        
        # Decoding with target and encoder output
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        
        # Projecting to vocabulary
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
        # Encoding the source
        encoder_output, src_mask = self.encoder(src, None)
        
        batch_size = encoder_output.size(0)
        
        # Initializing target with start symbol - ensuring correct device
        ys = torch.ones(batch_size, 1).fill_(start_symbol).long().to(self.device)
        
        # Generating tokens one by one
        for i in range(max_len-1):
            # Creating mask for target
            tgt_mask = self.decoder.create_target_mask(ys)
            
            # Decoding current sequence
            decoder_output = self.decoder(ys, encoder_output, src_mask, tgt_mask)
            
            # Getting next token (last position only)
            logits = self.projection(decoder_output)
            next_word = logits[:, -1].argmax(dim=-1, keepdim=True)
            
            # Appending to target sequence
            ys = torch.cat([ys, next_word], dim=1)
            
            # Stopping if all sequences have end token
            if (next_word == end_symbol).all():
                break
                
        return ys

def get_language_tokenizer(config):
    """
    Get the appropriate tokenizer for the language pair specified in config.
    Prioritizes language-specific tokenizer if specified in config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        tokenizer: HuggingFace tokenizer appropriate for the language pair
    """
    # Getting the default tokenizer name from config
    default_tokenizer_name = config["tokenizer_name"]
    tokenizer_name = default_tokenizer_name
    fallback_tokenizer_name = None
    second_fallback_tokenizer_name = "google/byt5-small"  # Universal fallback that works with any language
    
    # Checking if there's a language-specific tokenizer
    lang_pair = config.get("lang_pair", "en-ja")  # Defaulting to en-ja
    if lang_pair in config["language_pairs"]:
        lang_config = config["language_pairs"][lang_pair]
        # Using language-specific tokenizer if specified
        if "tokenizer_name" in lang_config:
            tokenizer_name = lang_config["tokenizer_name"]
            print(f"Using language-specific tokenizer for {lang_pair}: {tokenizer_name}")
            
            # Also getting language-specific fallback tokenizer if available
            if "fallback_tokenizer_name" in lang_config:
                fallback_tokenizer_name = lang_config["fallback_tokenizer_name"]
                
            # Getting second fallback tokenizer if available
            if "second_fallback_tokenizer_name" in lang_config:
                second_fallback_tokenizer_name = lang_config["second_fallback_tokenizer_name"]
    
    # For Japanese, using special handling to ensure proper tokenization
    if lang_pair == "en-ja" or lang_pair == "ja-en":
        try:
            # Trying the Japanese character-based tokenizer first (lighter dependencies)
            print("Attempting to load Japanese character-based tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-char')
            print("Successfully loaded Japanese character-based tokenizer")
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            return tokenizer
        except Exception as e:
            print(f"Error loading Japanese character tokenizer: {e}")
            try:
                # Trying the Japanese word-based tokenizer next
                print("Attempting to load Japanese word-based tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
                print("Successfully loaded Japanese word-based tokenizer")
                if tokenizer.pad_token is None:
                    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                return tokenizer
            except Exception as e2:
                print(f"Error loading Japanese word tokenizer: {e2}")
                print("Falling back to specified tokenizers...")
    
    # Trying to load the selected tokenizer
    try:
        # Loading the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        return tokenizer
    except Exception as e:
        # If we have a language-specific fallback tokenizer, trying that next
        if fallback_tokenizer_name:
            print(f"Error loading primary language-specific tokenizer '{tokenizer_name}': {e}")
            print(f"Trying first fallback tokenizer: {fallback_tokenizer_name}")
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(fallback_tokenizer_name)
                if tokenizer.pad_token is None:
                    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                return tokenizer
            except Exception as e2:
                print(f"Error loading first fallback tokenizer: {e2}")
                print(f"Trying second fallback tokenizer: {second_fallback_tokenizer_name}")
                
                try:
                    tokenizer = AutoTokenizer.from_pretrained(second_fallback_tokenizer_name)
                    if tokenizer.pad_token is None:
                        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                    return tokenizer
                except Exception as e3:
                    print(f"Error loading second fallback tokenizer: {e3}")
                    print(f"Falling back to default tokenizer: {default_tokenizer_name}")
        else:
            # If there's no fallback tokenizer and we were using a language-specific tokenizer
            if tokenizer_name != default_tokenizer_name:
                print(f"Error loading language-specific tokenizer '{tokenizer_name}': {e}")
                print(f"Trying universal fallback tokenizer: {second_fallback_tokenizer_name}")
                
                try:
                    tokenizer = AutoTokenizer.from_pretrained(second_fallback_tokenizer_name)
                    if tokenizer.pad_token is None:
                        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                    return tokenizer
                except Exception as e2:
                    print(f"Error loading universal fallback tokenizer: {e2}")
                    print(f"Falling back to default tokenizer: {default_tokenizer_name}")
        
        # As a last resort, loading the default tokenizer
        tokenizer = AutoTokenizer.from_pretrained(default_tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        return tokenizer

def build_transformer(config=None):
    """
    Build a transformer model from a configuration dictionary
    
    Args:
        config: Configuration dictionary, if None will use default from config.py
        
    Returns:
        Transformer model
    """
    # If no config provided, getting default from config.py
    if config is None:
        from config import get_config
        config = get_config()
    
    if 'device' not in config:
        config['device'] = get_device_config()
        
    target_device = config['device']
    print(f"Building transformer for device: {target_device}")
    
    try:
        tokenizer = get_language_tokenizer(config)
        vocab_size = len(tokenizer)
        
        # Getting max_len for positional encoding - default to max_seq_len if not specified
        max_len_pe = config.get("max_len_pe", config["max_seq_len"])
    
        # Building encoder
        encoder = Encoder(
            d_model=config["d_model"],
            vocab_size=vocab_size,
            seq_len=config["max_seq_len"],
            n_layers=config["num_layers"],
            h=config["num_heads"],
            d_ff=config["d_ff"],
            dropout=config["dropout"],
            tokenizer_name=tokenizer.name_or_path,  # Using actual tokenizer name
            max_len_pe=max_len_pe
        )
        
        # Building decoder
        decoder = Decoder(
            d_model=config["d_model"],
            vocab_size=vocab_size,
            seq_len=config["max_seq_len"],
            n_layers=config["num_layers"],
            h=config["num_heads"],
            d_ff=config["d_ff"],
            dropout=config["dropout"],
            max_len_pe=max_len_pe
        )
        
        # Building projection layer
        projection = ProjectionLayer(config["d_model"], vocab_size)
        
        # Building transformer
        transformer = Transformer(encoder, decoder, projection)
        
        # Initializing parameters
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Moving model to the configured device
        return transformer.to(target_device)
    except Exception as e:
        print(f"Error building transformer on {target_device}: {e}")
        print("Attempting to build on CPU and then move to target device")
        
        # Trying to build on CPU first
        cpu_device = torch.device('cpu')
        
        # Getting tokenizer to determine vocab size
        tokenizer = get_language_tokenizer(config)
        vocab_size = len(tokenizer)
        
        # Getting max_len for positional encoding - default to max_seq_len if not specified
        max_len_pe = config.get("max_len_pe", config["max_seq_len"])
        
        # Forcing CPU device temporarily
        old_device = device
        globals()['device'] = cpu_device
        
        # Building encoder on CPU
        encoder = Encoder(
            d_model=config["d_model"],
            vocab_size=vocab_size,
            seq_len=config["max_seq_len"],
            n_layers=config["num_layers"],
            h=config["num_heads"],
            d_ff=config["d_ff"],
            dropout=config["dropout"],
            tokenizer_name=tokenizer.name_or_path,
            max_len_pe=max_len_pe
        )
        
        # Building decoder on CPU
        decoder = Decoder(
            d_model=config["d_model"],
            vocab_size=vocab_size,
            seq_len=config["max_seq_len"],
            n_layers=config["num_layers"],
            h=config["num_heads"],
            d_ff=config["d_ff"],
            dropout=config["dropout"],
            max_len_pe=max_len_pe
        )
        
        # Building projection layer on CPU
        projection = ProjectionLayer(config["d_model"], vocab_size)
        
        # Building transformer on CPU
        transformer = Transformer(encoder, decoder, projection)
        
        # Initializing parameters
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Restore original device
        globals()['device'] = old_device
        
        # Try to move to target device
        try:
            print(f"Moving transformer from CPU to {target_device}")
            return transformer.to(target_device)
        except Exception as e2:
            print(f"Failed to move to {target_device}: {e2}")
            print("Keeping transformer on CPU")
            return transformer

if __name__ == "__main__":
    # Building model
    model = build_transformer()
    device = get_device_config()
    
    # Testing with some example data
    src_texts = ["Hello, world!", "This is a test."]
    tgt_tokens = torch.ones(2, 5).long() * 101  # Example target tokens
    tgt_tokens = tgt_tokens.to(device)  # Moving to the correct device
    
    # Running forward pass
    output = model(src_texts, tgt_tokens)
    
    # Generating sequence
    generated = model.generate(src_texts)
    
    print(f"Output shape: {output.shape}")
    print(f"Generated shape: {generated.shape}")
    print("Test successful!") 