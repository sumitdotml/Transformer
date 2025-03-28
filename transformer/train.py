import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import datetime
import time
import json
from pathlib import Path
import argparse

from config import get_config, get_weights_file_path, latest_weights_file_path, get_device_config
from simplified_model import build_transformer

device = get_device_config()

class TranslationDataset(Dataset):
    """Custom dataset for machine translation"""
    
    def __init__(self, dataset, tokenizer, src_lang, tgt_lang, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        # Getting source and target texts
        src_text = self.dataset[idx]["translation"][self.src_lang]
        tgt_text = self.dataset[idx]["translation"][self.tgt_lang]
        
        # Tokenizing target text (need token ids here for input to decoder)
        tgt_tokens = self.tokenizer(
            tgt_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "src_text": src_text,
            "tgt_text": tgt_text,
            "tgt_tokens": tgt_tokens["input_ids"].squeeze()
        }

def train_epoch(model, data_loader, optimizer, criterion, device, clip_grad=1.0):
    """Train model for one epoch with gradient clipping"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        # Getting source and target
        src_texts = batch["src_text"]
        tgt_tokens = batch["tgt_tokens"].to(device)
        
        # Shifting targets for teacher forcing 
        # (model should predict next token given previous tokens)
        input_tgt = tgt_tokens[:, :-1]
        output_tgt = tgt_tokens[:, 1:]
        
        # Forward pass
        output = model(src_texts, input_tgt)
        
        # Reshaping output and target for loss calculation
        output = output.reshape(-1, output.size(-1))
        output_tgt = output_tgt.reshape(-1)
        
        # Calculating loss
        loss = criterion(output, output_tgt)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(data_loader)

def evaluate(model, data_loader, criterion, device):
    """Evaluate model on validation dataset"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Getting source and target
            src_texts = batch["src_text"]
            tgt_tokens = batch["tgt_tokens"].to(device)
            
            # Shifting targets for teacher forcing
            input_tgt = tgt_tokens[:, :-1]
            output_tgt = tgt_tokens[:, 1:]
            
            # Forward pass
            output = model(src_texts, input_tgt)
            
            # Reshaping output and target for loss calculation
            output = output.reshape(-1, output.size(-1))
            output_tgt = output_tgt.reshape(-1)
            
            # Calculating loss
            loss = criterion(output, output_tgt)
            total_loss += loss.item()
            
    return total_loss / len(data_loader)

def translate_example(model, tokenizer, text, tgt_lang, src_lang="en", max_len=100):
    """Translate a single text example"""
    model.eval()
    
    # Generating with the model
    with torch.no_grad():
        generated_ids = model.generate([text], max_len=max_len)
    
    # Decoding generated token IDs to text
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    print(f"{src_lang}: {text}")
    print(f"{tgt_lang}: {generated_text[0]}")
    return generated_text[0]

def get_timestamp():
    """Get current timestamp string for file naming"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def save_training_plot(train_losses, val_losses, output_dir, timestamp=None):
    """Save training history plot with timestamp"""
    if timestamp is None:
        timestamp = get_timestamp()
        
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename = f"training_history_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print(f"Training history saved to {filepath}")
    
    return filepath

def save_training_state(config, model, optimizer, train_losses, val_losses, epoch, 
                        output_dir, timestamp=None, is_best=False):
    """Save model checkpoint and training state"""
    if timestamp is None:
        timestamp = get_timestamp()
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Saving model weights using both timestamp approach and the project's standard path
    prefix = "best_" if is_best else ""
    
    # Timestamp-based path (for organization)
    model_filename = f"{prefix}model_{timestamp}_epoch_{epoch}.pth"
    model_path = os.path.join(output_dir, model_filename)
    
    # Also saving to the standard path from config.py (for compatibility)
    standard_model_path = get_weights_file_path(config, f"{epoch}")
    
    # Create standard weights directory if it doesn't exist
    model_folder = config["model_folder"]
    os.makedirs(model_folder, exist_ok=True)
    
    # Saving model to both locations
    torch.save(model.state_dict(), model_path)
    torch.save(model.state_dict(), standard_model_path)
    
    # Saving optimizer state and training progress
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),  # Adding model state to checkpoint for compatibility
        'optimizer': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': config
    }
    
    checkpoint_filename = f"{prefix}checkpoint_{timestamp}_epoch_{epoch}.pt"
    checkpoint_path = os.path.join(output_dir, checkpoint_filename)
    torch.save(checkpoint, checkpoint_path)
    
    print(f"Model saved to {model_path}")
    print(f"Model also saved to standard path: {standard_model_path}")
    print(f"Checkpoint saved to {checkpoint_path}")
    
    return model_path, checkpoint_path, standard_model_path

def load_training_state(checkpoint_path, model, optimizer):
    """Load model checkpoint and training state"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Loading model weights (handle both formats)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Trying to load the model from its standard path
        config = checkpoint['config']
        epoch = checkpoint['epoch']
        model_path = get_weights_file_path(config, f"{epoch}")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            raise FileNotFoundError(f"Could not find model weights at {model_path}")
    
    # Loading optimizer state
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Returning training progress
    return (
        checkpoint['epoch'],
        checkpoint['train_losses'],
        checkpoint['val_losses']
    )

def main():
    # Creating the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, help='Path to a checkpoint file to resume training from.')
    parser.add_argument('--lang-pair', type=str, default='en-ja', help='Language pair to train (e.g., en-ja, en-de).')
    parser.add_argument('--dry-run', action='store_true', help='Initialize but don\'t train (for testing).')
    parser.add_argument('--test-dataset', action='store_true', help='Only test dataset loading without training.')
    parser.add_argument('--limit', type=int, help='Limit the number of examples to use (for faster testing).')
    args = parser.parse_args()
    
    # Getting the device
    device = get_device_config()
    print(f"Training will use device: {device}")
    
    # Getting configuration
    config = get_config()
    
    # Updating config with language-specific configuration if available
    if args.lang_pair in config["language_pairs"]:
        lang_config = config["language_pairs"][args.lang_pair]
        print(f"Using language-specific configuration for {args.lang_pair}")
        
        # Updating config with language-specific values
        for key, value in lang_config.items():
            config[key] = value
    else:
        print(f"No specific configuration found for {args.lang_pair}, using defaults")
    
    # Storing the language pair in the config
    config["lang_pair"] = args.lang_pair
    
    # If dry run, exit after initialization
    if args.dry_run:
        print(f"Dry run successful for language pair: {args.lang_pair}")
        print(f"Configuration: {config}")
        return
    
    # Parsing language pair
    src_lang, tgt_lang = args.lang_pair.split("-")
    print(f"Training {src_lang} to {tgt_lang} translation model")
    
    # Creating output directories
    timestamp = get_timestamp()
    
    # Using outputs folder for all outputs
    root_dir = config["root_dir"]
    outputs_dir = os.path.join(root_dir, "outputs")
    training_output_dir = os.path.join(outputs_dir, "training")
    
    # Creating timestamped run directory
    output_dir = os.path.join(training_output_dir, f"{timestamp}_{args.lang_pair}")
    weights_dir = os.path.join(output_dir, "weights")
    plots_dir = os.path.join(output_dir, "plots")
    logs_dir = os.path.join(output_dir, "logs")
    
    for directory in [outputs_dir, training_output_dir, output_dir, weights_dir, plots_dir, logs_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Creating a copy of the config for saving to file
    # Stripping absolute paths to avoid leaking system details
    config_for_saving = config.copy()
    for key in config_for_saving:
        if isinstance(config_for_saving[key], str) and os.path.isabs(config_for_saving[key]):
            # Replacing absolute paths with relative ones
            config_for_saving[key] = os.path.basename(config_for_saving[key])
        elif isinstance(config_for_saving[key], dict):
            # Handling nested dictionaries like language_pairs
            for sub_key, sub_value in config_for_saving[key].items():
                if isinstance(sub_value, dict):
                    for k, v in sub_value.items():
                        if isinstance(v, str) and os.path.isabs(v):
                            config_for_saving[key][sub_key][k] = os.path.basename(v)
    
    # Saving config to output directory
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_for_saving, f, indent=2)
    
    # Loading dataset
    print("Loading dataset...")
    try:
        # First trying the opus-100 dataset that has actual English-Japanese pairs
        split = "train[:{}]".format(args.limit) if args.limit else "train"
        dataset = load_dataset("Helsinki-NLP/opus-100", "en-ja", split=split)
        print(f"Loaded {len(dataset)} en-ja translation examples from opus-100")
    except Exception as e:
        print(f"Error loading opus-100 dataset: {e}")
        try:
            # Then trying opus_books
            split = "train[:{}]".format(args.limit) if args.limit else "train"
            dataset = load_dataset("Helsinki-NLP/opus_books", "en-ja", split=split)
            print(f"Loaded {len(dataset)} translation examples from opus_books")
        except Exception as e1:
            print(f"Error loading Helsinki-NLP/opus_books: {e1}")
            try:
                # Fallback to JParaCrawl dataset
                limit = args.limit if args.limit else 100000
                split = f"train[:{limit}]"
                dataset = load_dataset("yuriseki/JParaCrawl", split=split)
                print(f"Loaded {len(dataset)} translation examples from JParaCrawl")
            except Exception as e2:
                print(f"Error loading JParaCrawl dataset: {e2}")
                try:
                    # Trying WMT19 dataset
                    limit = args.limit if args.limit else 20000
                    split = f"train[:{limit}]"
                    dataset = load_dataset("wmt19", "ja-en", split=split)
                    # Swapping source and target to get en-ja
                    dataset = dataset.map(lambda x: {
                        "translation": {
                            "en": x["translation"]["ja"],
                            "ja": x["translation"]["en"]
                        }
                    })
                    print(f"Loaded {len(dataset)} translation examples from wmt19")
                except Exception as e3:
                    print(f"Failed to load any Japanese datasets: {e3}")
                    raise RuntimeError("Could not load any suitable dataset for training. Please check your internet connection and dataset availability.")
    
    # Splitting dataset into train and validation
    train_size = int(0.9 * len(dataset))
    # No need to store val_size as it's not used directly
    
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, len(dataset)))
    
    print(f"Training set: {len(train_dataset)} examples")
    print(f"Validation set: {len(val_dataset)} examples")
    
    # Exiting early if only testing dataset loading
    if args.test_dataset:
        print("Dataset loading test successful!")
        print(f"Example English: {train_dataset[0]['translation']['en']}")
        print(f"Example Japanese: {train_dataset[0]['translation']['ja']}")
        return
    
    # Building model
    print("Building model...")
    model = build_transformer(config).to(device)
    
    # Checking if there's a previously trained model we can continue from
    latest_weights = latest_weights_file_path(config)
    if latest_weights and os.path.exists(latest_weights):
        print(f"Found previously trained weights at {latest_weights}")
        try:
            model.load_state_dict(torch.load(latest_weights, map_location=device))
            print(f"Successfully loaded weights from {latest_weights}")
            
            # Saving a copy of the loaded weights to our new training run
            pretrained_path = os.path.join(weights_dir, f"pretrained_{timestamp}.pth")
            torch.save(model.state_dict(), pretrained_path)
            print(f"Saved a copy of loaded weights to {pretrained_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Continuing with freshly initialized model")
    
    # Getting tokenizer from the model
    tokenizer = model.encoder.tokenizer
    
    # Creating datasets
    train_dataset_obj = TranslationDataset(
        train_dataset,
        tokenizer, 
        src_lang=src_lang, 
        tgt_lang=tgt_lang,
        max_length=config["max_seq_len"]
    )
    
    val_dataset_obj = TranslationDataset(
        val_dataset,
        tokenizer, 
        src_lang=src_lang, 
        tgt_lang=tgt_lang,
        max_length=config["max_seq_len"]
    )
    
    # Creating data loaders
    train_loader = DataLoader(
        train_dataset_obj, 
        batch_size=config["batch_size"], 
        shuffle=True,
        num_workers=config["num_workers"]  # Use config value
    )
    
    val_loader = DataLoader(
        val_dataset_obj, 
        batch_size=config["batch_size"], 
        shuffle=False,
        num_workers=config["num_workers"]  # Use config value
    )
    
    # Defining optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["learning_rate"],
        betas=(0.9, 0.98),
        eps=1e-9
    )
    criterion = nn.NLLLoss(ignore_index=tokenizer.pad_token_id)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2,
        verbose=True
    )
    
    # Training loop variables
    num_epochs = config["num_epochs"]
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    start_epoch = 0
    
    # Resuming from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from checkpoint: {args.resume}")
        try:
            # Loading checkpoint
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Loading model weights
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Trying to find the model file
                config_from_checkpoint = checkpoint.get('config', config)
                epoch = checkpoint.get('epoch', 0)
                model_path = get_weights_file_path(config_from_checkpoint, f"{epoch}")
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, map_location=device))
                else:
                    print(f"Warning: Could not find model weights at {model_path}")
                    print("Continuing with fresh model weights")
            
            # Loading optimizer state
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            
            # Resuming training progress
            if 'train_losses' in checkpoint:
                train_losses = checkpoint['train_losses']
            if 'val_losses' in checkpoint:
                val_losses = checkpoint['val_losses']
                # Updating best validation loss
                if val_losses:
                    best_val_loss = min(min(val_losses), best_val_loss)
            
            # Resuming from the next epoch
            start_epoch = checkpoint.get('epoch', 0)
            print(f"Resuming from epoch {start_epoch+1}")
            
            # Saving a copy of the checkpoint to my new training run
            copy_path = os.path.join(weights_dir, f"resumed_from_{os.path.basename(args.resume)}")
            torch.save(checkpoint, copy_path)
            print(f"Saved a copy of the checkpoint to {copy_path}")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting fresh training run")
    
    # Log file for tracking progress
    log_file = os.path.join(logs_dir, f"training_log_{timestamp}.txt")
    
    print(f"Starting training for {num_epochs} epochs, from epoch {start_epoch+1}...")
    
    # If I've loaded saved losses, plotting the progress so far
    if train_losses and val_losses:
        initial_plot_path = save_training_plot(train_losses, val_losses, plots_dir, timestamp + "_resumed")
        print(f"Plotted training progress so far to {initial_plot_path}")
    
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, clip_grad=config["clip_grad"])
        train_losses.append(train_loss)
        
        # Evaluating
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Updating learning rate
        scheduler.step(val_loss)
        
        # Calculating epoch time
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        print(f"Epoch time: {epoch_time:.2f} seconds")
        
        # Checking if this is the best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Translating example for demonstration
        if epoch % 1 == 0:
            example = train_dataset[0]["translation"]["en"]
            translate_example(model, tokenizer, example, tgt_lang)
        
        # Saving model checkpoint
        save_training_state(
            config, model, optimizer, train_losses, val_losses, 
            epoch+1, weights_dir, timestamp, is_best
        )
        
        # Saving training plot every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_training_plot(train_losses, val_losses, plots_dir, timestamp)
        
        # Logging progress
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch+1}/{num_epochs}, "
                    f"Train loss: {train_loss:.4f}, "
                    f"Val loss: {val_loss:.4f}, "
                    f"Time: {epoch_time:.2f}s\n")
        
        # Early stopping
        if patience_counter >= config["patience"]:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Saving final training plot
    final_plot_path = save_training_plot(train_losses, val_losses, plots_dir, timestamp)
    
    # Loading the best model for final evaluation
    best_model_path = os.path.join(weights_dir, f"best_model_{timestamp}_epoch_*.pth")
    best_model_files = list(Path(weights_dir).glob(f"best_model_{timestamp}_epoch_*.pth"))
    
    if best_model_files:
        best_model_path = str(best_model_files[0])
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from {best_model_path}")
    
    # Demonstrating translation with the best model
    print("\nTranslation examples with best model:")
    for i in range(5):
        example = val_dataset[i]["translation"]["en"]
        translate_example(model, tokenizer, example, tgt_lang)
    
    # Saving final model using both approaches
    final_model_path = os.path.join(weights_dir, f"final_model_{timestamp}.pth")
    
    # Creating model folder from config if it doesn't exist
    model_folder = config["model_folder"]
    os.makedirs(model_folder, exist_ok=True)
    
    final_standard_path = get_weights_file_path(config, "final")
    
    torch.save(model.state_dict(), final_model_path)
    torch.save(model.state_dict(), final_standard_path)
    print(f"Final model saved to {final_model_path}")
    print(f"Final model also saved to standard path: {final_standard_path}")
    print(f"Final training plot saved to {final_plot_path}")
    
    print(f"Training completed! All outputs saved to {output_dir}")

if __name__ == "__main__":
    main() 