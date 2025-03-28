from pathlib import Path
import os

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

def get_config():
    """
    Returns default configuration parameters for the transformer model.
    This is where we can adjust these values for our specific needs.
    """
    return {
        # Model architecture
        "d_model": 512,        # Embedding dimension
        "num_heads": 8,        # Number of attention heads
        "num_layers": 6,       # Number of encoder/decoder layers
        "d_ff": 2048,          # Feed-forward dimension
        "dropout": 0.1,        # Dropout rate
        "max_seq_len": 512,    # Maximum sequence length
        "tokenizer_name": "bert-base-uncased",  # Default tokenizer
        
        # Training
        "batch_size": 8,
        "num_epochs": 50,
        "learning_rate": 1e-4,
        "patience": 5,         # Early stopping patience
        "clip_grad": 1.0,      # Gradient clipping value
        "num_workers": 2,      # Data loader workers
        
        # Paths - using absolute paths for clarity
        "root_dir": str(PROJECT_ROOT),
        "model_folder": str(PROJECT_ROOT / "outputs" / "weights"),
        "model_basename": "transformer_",
        "preload": "latest",
        "demo_dir": str(PROJECT_ROOT / "outputs" / "demo"),
        "demo_basename": "transformer_demo_",
        
        # Language-specific configurations
        "language_pairs": {
            "en-ja": {
                "num_epochs": 50,
                "batch_size": 16,
                "learning_rate": 1e-4,
                "dropout": 0.2,
                "patience": 5,
                "tokenizer_name": "cl-tohoku/bert-base-japanese-char",  # Character-based Japanese tokenizer (no extra dependencies)
                "fallback_tokenizer_name": "cl-tohoku/bert-base-japanese",  # MeCab-based tokenizer (requires fugashi)
                "second_fallback_tokenizer_name": "google/byt5-small"  # Universal tokenizer that works with any language
            }
        },
        
        # Device configuration
        "use_cuda": True,      # Use CUDA if available
        "use_mps": True        # Use MPS if available (for Apple Silicon)
    }

def get_weights_file_path(config, epoch: str):
    """Get the path to a specific weights file by epoch"""
    model_folder = config['model_folder']
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return os.path.join(model_folder, model_filename)

def latest_weights_file_path(config):
    """Find the latest weights file in the weights folder"""
    model_folder = config['model_folder']
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])

def get_device_config():
    """Get the appropriate device based on availability and configuration"""
    import torch
    
    # Default configuration
    config = get_config()
    
    # First trying CUDA
    if config["use_cuda"] and torch.cuda.is_available():
        return torch.device("cuda")
    
    # Then trying MPS (Apple Silicon GPU)
    elif config["use_mps"] and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            # Creating a small test tensor to verify MPS is working properly
            test_tensor = torch.zeros(1, device="mps")
            test_tensor = test_tensor + 1  # Simple operation to test device
            return torch.device("mps")
        except Exception as e:
            return torch.device("cpu")
    
    # Fallback to CPU
    else:
        return torch.device("cpu") 