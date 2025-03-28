# Transformer Training Guide

This guide explains how to train the transformer model for English to Japanese translation.

## Quick Start

The easiest way to start training is to use the provided script:

```bash
# Make the script executable
chmod +x train.sh

# Run the training script
./train.sh
```

This will:

1. Create a virtual environment if needed
2. Install required dependencies
3. Start the training process

### Resuming Training

We can resume training from a previously saved checkpoint:

```bash
./train.sh --resume path/to/checkpoint_file.pt
```

This allows us to:

- Continue training after interruption
- Start from a pre-trained model
- Fine-tune on new data

The checkpoint contains all necessary state:

- Model weights
- Optimizer state
- Training history
- Configuration

### Selecting Language Pairs

We can train on different language pairs:

```bash
# Train English to Japanese (default)
./train.sh

# Train English to German
./train.sh --lang-pair en-de

# Train any other supported language pair
./train.sh --lang-pair source-target
```

The script will automatically use the language-specific configuration defined in `config.py`.

### Testing and Development Options

For testing and development, several flags are available:

```bash
# Test only dataset loading without training (useful for dataset verification)
python transformer/train.py --test-dataset

# Limit the dataset size for faster experimentation
python transformer/train.py --limit 1000

# Initialize model and configuration without training (dry run)
python transformer/train.py --dry-run
```

## Dataset Handling

The training script automatically fetches and uses datasets from Hugging Face, with multiple fallback options:

1. **Helsinki-NLP/opus-100** (primary): 1 million high-quality English-Japanese pairs
2. **Helsinki-NLP/opus_books**: English-Japanese book translations
3. **yuriseki/JParaCrawl**: Web-crawled parallel corpus
4. **WMT19**: News translations with language swapping

The prioritization ensures we always get the best available dataset. Example:

```bash
# Use the opus-100 dataset with 1000 examples
python transformer/train.py --limit 1000

# View sample data from the dataset
python transformer/train.py --test-dataset
```

## Training Details

The training script (`transformer/train.py`) implements the following features:

- Training on English-Japanese translation datasets
- Automated dataset downloading from Hugging Face
- Multiple dataset fallbacks if the primary dataset isn't available
- Gradient clipping to prevent exploding gradients
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping to prevent overfitting
- Checkpointing of the best model based on validation loss
- Timestamped output files:
  - Model checkpoints
  - Training plots
  - Detailed logs

## Output Structure

All outputs are organized with absolute paths to prevent confusion:

```
/project_root/
├── outputs/
│   ├── demo/                  # Demo outputs
│   │   ├── training_history_timestamp.png
│   │   └── transformer_model_timestamp.pth
│   └── weights/               # Training weights (not used by demo)
└── transformer/
    └── training_output/       # Training run outputs
        └── timestamp_lang-pair/
            ├── config.json
            ├── logs/
            │   └── training_log_timestamp.txt
            ├── plots/
            │   └── training_history_timestamp.png
            └── weights/
                ├── best_model_timestamp_epoch_10.pth
                └── ...
```

This organization ensures:

1. Demo outputs stay separate from training weights
2. Each training run has its own timestamped directory
3. All paths are consistent regardless of where scripts are run from

## Directory Structure

The project uses consistent absolute paths to ensure files are always saved in the correct locations:

- **Demo outputs**: `/project_root/outputs/demo/`
- **Training weights**: `/project_root/outputs/weights/`
- **Training run data**: `/project_root/transformer/training_output/{timestamp}_{lang_pair}/`

These paths are configured in `config.py` and automatically resolved relative to the project root:

```python
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

config = {
    # Paths - using absolute paths for clarity
    "root_dir": str(PROJECT_ROOT),
    "model_folder": str(PROJECT_ROOT / "outputs" / "weights"),
    "demo_dir": str(PROJECT_ROOT / "outputs" / "demo"),
    ...
}
```

## Configuration

We can modify the training configuration in the `config.py` file. The system includes:

1. **General configuration settings** applicable to all training runs
2. **Language-specific configurations** for different language pairs
3. **Device configuration** to handle different hardware (CUDA, MPS, CPU)

```python
# General configuration example
config = {
    "d_model": 512,
    "batch_size": 8,
    "learning_rate": 1e-4,
    ...
}

# Language-specific configuration example
config["language_pairs"] = {
    "en-ja": {
        "num_epochs": 20,
        "batch_size": 16,
        ...
    },
    "en-de": {
        "num_epochs": 15,
        "batch_size": 32,
        ...
    }
}
```

To add a new language pair, simply add a new entry to the `language_pairs` dictionary.

## Device Handling

The training code automatically selects the appropriate device based on availability:

1. **CUDA** (NVIDIA GPUs) - Used if available and enabled
2. **MPS** (Apple Silicon) - Used if available, enabled, and CUDA isn't available
3. **CPU** - Fallback option

We can configure device preferences in `config.py`:

```python
config = {
    ...
    "use_cuda": True,  # Use CUDA if available
    "use_mps": True    # Use MPS if available (for Apple Silicon)
}
```

## Manual Training

If we prefer not to use the script, we can manually run:

```bash
python transformer/train.py

# With custom options
python transformer/train.py --lang-pair en-ja --limit 10000
```

## Common Command Line Options

The training script supports several command line arguments:

| Flag             | Description                       | Example                          |
| ---------------- | --------------------------------- | -------------------------------- |
| `--resume`       | Resume training from a checkpoint | `--resume path/to/checkpoint.pt` |
| `--lang-pair`    | Language pair to train            | `--lang-pair en-ja`              |
| `--dry-run`      | Initialize without training       | `--dry-run`                      |
| `--test-dataset` | Test dataset loading only         | `--test-dataset`                 |
| `--limit`        | Limit dataset size                | `--limit 1000`                   |

## Using a Trained Model

After training, we can use the trained model for inference in two ways:

### 1. Using the timestamp-based paths:

```python
from transformer.simplified_model import build_transformer
from transformer.config import get_config, get_device_config
import torch

# Load configuration
config = get_config()

# Get device
device = get_device_config()

# Build model
model = build_transformer(config).to(device)

# Load trained weights from a specific training run
model.load_state_dict(torch.load("transformer/training_output/20240327_123456/weights/best_model_20240327_123456_epoch_10.pth"))

# Generate translation
english_text = "Hello, how are you today?"
japanese_translation = model.translate(english_text, max_len=100)
print(japanese_translation)
```

### 2. Using the standard paths from config:

```python
from transformer.simplified_model import build_transformer
from transformer.config import get_config, get_device_config, get_weights_file_path
import torch

# Load configuration
config = get_config()

# Get device
device = get_device_config()

# Build model
model = build_transformer(config).to(device)

# Get the standard path for the model weights
weights_path = get_weights_file_path(config, "final")  # Or use "10" for epoch 10

# Load trained weights
model.load_state_dict(torch.load(weights_path))

# Generate translation
english_text = "Hello, how are you today?"
japanese_translation = model.translate(english_text, max_len=100)
print(japanese_translation)
```

## Demo Mode

For demonstration purposes, we can use the demo script:

```bash
python transformer/demo.py
```

The demo script:

1. Uses the Helsinki-NLP/opus-100 dataset with actual Japanese examples
2. Trains a small model for 2 epochs
3. Saves outputs (plots and model) only to the demo directory
4. Shows example translations

Demo outputs are kept separate from training files and go to `/project_root/outputs/demo/`.
