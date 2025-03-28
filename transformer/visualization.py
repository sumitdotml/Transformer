import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from transformers import AutoTokenizer
import seaborn as sns
from config import get_config, get_device_config
from simplified_model import build_transformer

def visualize_attention(attention_weights, tokens, layer=0, head=0, cmap='viridis', title=None):
    """
    Visualize the attention weights for a specific layer and head.
    
    Args:
        attention_weights: Tensor of shape [n_layers, n_heads, seq_len, seq_len]
                          or [n_heads, seq_len, seq_len]
        tokens: List of tokens corresponding to the sequence
        layer: Layer to visualize (if attention_weights contains layer dimension)
        head: Attention head to visualize
        cmap: Matplotlib colormap to use
        title: Title for the plot
    """
    if attention_weights is None:
        print("No attention weights provided")
        return
    
    # Handle different input shapes
    if len(attention_weights.shape) == 4:  # [n_layers, n_heads, seq_len, seq_len]
        attn = attention_weights[layer, head].cpu().numpy()
    elif len(attention_weights.shape) == 3:  # [n_heads, seq_len, seq_len]
        attn = attention_weights[head].cpu().numpy()
    else:
        raise ValueError(f"Unexpected attention weights shape: {attention_weights.shape}")
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(
        attn, 
        xticklabels=tokens, 
        yticklabels=tokens, 
        ax=ax, 
        cmap=cmap,
        annot=False,
        square=True,
        cbar_kws={"shrink": 0.8}
    )
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Attention Weights (Layer {layer}, Head {head})")
    
    # Adjust font size
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.tight_layout()
    return fig

def get_token_representations(model, text, tokenizer=None):
    """
    Get token representations at various layers of the model.
    
    Args:
        model: The transformer model
        text: Input text
        tokenizer: Optional tokenizer (if not provided, uses model's encoder tokenizer)
        
    Returns:
        Dictionary with token representations
    """
    # Use model's tokenizer if not provided
    if tokenizer is None:
        tokenizer = model.encoder.tokenizer
    
    # Tokenize input
    encoding = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    )
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    
    with torch.no_grad():
        # Forward pass through encoder
        encoder_output, _ = model.encoder([text])
        
        # Create a simple target sequence for decoder input
        target_seq = torch.ones((1, 5)).long() * tokenizer.cls_token_id
        
        # Forward pass through decoder
        decoder_output = model.decoder(
            target_seq,
            encoder_output,
            None  # No source mask
        )
    
    return {
        "tokens": tokens,
        "encoder_output": encoder_output[0].cpu().numpy(),
        "decoder_output": decoder_output[0].cpu().numpy()
    }

def visualize_token_embeddings(token_representations, dim_reduction='pca', n_components=2):
    """
    Visualize token embeddings using dimensionality reduction.
    
    Args:
        token_representations: Output from get_token_representations
        dim_reduction: Method for dimensionality reduction ('pca' or 'tsne')
        n_components: Number of components to reduce to
        
    Returns:
        Figure object
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    # Get tokens and embeddings
    tokens = token_representations["tokens"]
    embeddings = token_representations["encoder_output"]
    
    # Apply dimensionality reduction
    if dim_reduction.lower() == 'pca':
        reducer = PCA(n_components=n_components)
    elif dim_reduction.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {dim_reduction}")
    
    # Apply reduction
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if n_components == 2:
        # 2D plot
        ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
        
        # Add token labels
        for i, token in enumerate(tokens):
            ax.annotate(token, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))
            
        ax.set_title(f"Token Embeddings Visualization ({dim_reduction.upper()})")
        
    elif n_components == 3:
        # 3D plot
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            reduced_embeddings[:, 0], 
            reduced_embeddings[:, 1], 
            reduced_embeddings[:, 2]
        )
        
        # Add token labels
        for i, token in enumerate(tokens):
            ax.text(
                reduced_embeddings[i, 0], 
                reduced_embeddings[i, 1], 
                reduced_embeddings[i, 2],
                token
            )
            
        ax.set_title(f"Token Embeddings Visualization ({dim_reduction.upper()}, 3D)")
    
    plt.tight_layout()
    return fig

def visualize_positional_encoding(model, max_len=100):
    """
    Visualize the positional encoding patterns.
    
    Args:
        model: The transformer model
        max_len: Maximum sequence length to visualize
        
    Returns:
        Figure object
    """
    # Extract positional encoding
    pos_encoding = model.encoder.positional_encoding.pe[0, :max_len, :].detach().cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(
        pos_encoding,
        ax=ax,
        cmap='coolwarm',
        center=0,
        cbar_kws={"shrink": 0.8}
    )
    
    ax.set_title("Positional Encoding Patterns")
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Position")
    
    plt.tight_layout()
    return fig

def plot_attention_patterns(model, text, save_path=None):
    """
    Create and save a comprehensive visualization of attention patterns.
    
    Args:
        model: The transformer model
        text: Input text
        save_path: Path to save the figure
        
    Returns:
        None
    """
    # Tokenize the input
    tokenizer = model.encoder.tokenizer
    tokens = tokenizer.tokenize(text)
    
    # Add special tokens for display
    display_tokens = ['[CLS]'] + tokens + ['[SEP]']
    
    # Create dummy attention weights for visualization
    # In a real scenario, you would extract them from the model
    n_heads = 8
    seq_len = len(display_tokens)
    
    # Create random attention patterns for demonstration
    # This would be replaced with actual attention extraction in a real model
    dummy_attention = torch.rand(n_heads, seq_len, seq_len)
    
    # Create a 2x4 grid for 8 attention heads
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.flatten()
    
    for h in range(n_heads):
        sns.heatmap(
            dummy_attention[h].numpy(),
            xticklabels=display_tokens,
            yticklabels=display_tokens,
            ax=axs[h],
            cmap='viridis',
            annot=False,
            square=True,
            cbar_kws={"shrink": 0.8}
        )
        axs[h].set_title(f"Attention Head {h+1}")
        axs[h].set_xticklabels(display_tokens, rotation=90, fontsize=8)
        axs[h].set_yticklabels(display_tokens, fontsize=8)
    
    plt.suptitle(f"Attention Patterns for: '{text}'", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved attention visualization to {save_path}")
    
    return fig

def main():
    """
    Main function to demonstrate visualization capabilities.
    """
    # Get device using the centralized function
    device = get_device_config()
    
    # Load configuration
    config = get_config()
    
    # Build model
    model = build_transformer(config).to(device)
    
    # Example text
    text = "Transformers are neural network models that process sequential data."
    
    # Visualize attention patterns
    visualization_dir = "transformer/visualizations"
    if not os.path.exists(visualization_dir):
        os.makedirs(visualization_dir)
    save_path = os.path.join(visualization_dir, "attention_patterns.png")
    fig = plot_attention_patterns(model, text, save_path=save_path if not os.path.exists(save_path) else None)
    
    # Visualize positional encoding
    fig_pos = visualize_positional_encoding(model)
    fig_pos.savefig(os.path.join(visualization_dir, "positional_encoding.png") if not os.path.exists(os.path.join(visualization_dir, "positional_encoding.png")) else None)
    
    print("Visualizations generated successfully!")

if __name__ == "__main__":
    main() 