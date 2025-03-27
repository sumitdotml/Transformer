from pathlib import Path

def get_config():
    """
    Returns default configuration parameters for the transformer model.
    Feel free to adjust these values for your specific needs.
    """
    return {
        # Model architecture
        "d_model": 512,        # Embedding dimension
        "num_heads": 8,        # Number of attention heads
        "num_layers": 6,       # Number of encoder/decoder layers
        "d_ff": 2048,          # Feed-forward dimension
        "dropout": 0.1,        # Dropout rate
        
        # Sequence parameters
        "max_seq_len": 512,    # Maximum sequence length
        
        # Tokenizer
        "tokenizer_name": "bert-base-uncased",  # Default tokenizer
        
        # Training
        "batch_size": 8,
        "num_epochs": 20,
        "learning_rate": 1e-4,
        
        # Model saving
        "model_folder": "weights",
        "model_basename": "transformer_",
        "preload": "latest"
    }

def get_weights_file_path(config, epoch: str):
    """Get the path to a specific weights file by epoch"""
    model_folder = config['model_folder']
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    """Find the latest weights file in the weights folder"""
    model_folder = config['model_folder']
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1]) 