import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import datetime

# Import configuration and model building functions
from config import get_config, get_weights_file_path, get_device_config
from simplified_model import build_transformer

# Set device using the centralized function
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
        # Get source and target texts
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

def train_epoch(model, data_loader, optimizer, criterion, device):
    """Train model for one epoch"""
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
        # output: [batch, seq_len, vocab_size] -> [batch*seq_len, vocab_size]
        # target: [batch, seq_len] -> [batch*seq_len]
        output = output.reshape(-1, output.size(-1))
        output_tgt = output_tgt.reshape(-1)
        
        # Calculating loss
        loss = criterion(output, output_tgt)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
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

def translate_example(model, tokenizer, text, tgt_lang, src_lang="en", max_len=50):
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

def main():
    # Loading configuration
    config = get_config()
    
    # Set language pair for Japanese translation (to use Japanese-specific tokenizer)
    lang_pair = "en-ja"
    config["lang_pair"] = lang_pair
    
    # Apply language-specific configurations if available
    if lang_pair in config["language_pairs"]:
        lang_config = config["language_pairs"][lang_pair]
        print(f"Using language-specific configuration for {lang_pair}")
        
        # Update config with language-specific values
        for key, value in lang_config.items():
            config[key] = value
    
    # Loading dataset (real Japanese dataset)
    print("Loading dataset...")
    try:
        # First try the opus-100 dataset that has actual English-Japanese pairs
        dataset = load_dataset("Helsinki-NLP/opus-100", "en-ja", split="validation[:100]")
        print(f"Loaded {len(dataset)} en-ja translation examples from opus-100")
    except Exception as e:
        print(f"Error loading opus-100 dataset: {e}")
        try:
            # Try opus_books as a first fallback
            dataset = load_dataset("Helsinki-NLP/opus_books", "en-ja", split="train[:100]")
            print(f"Loaded {len(dataset)} en-ja translation examples from opus_books")
        except Exception as e2:
            print(f"Error loading opus_books dataset: {e2}")
            try:
                # Try JParaCrawl as a second fallback
                dataset = load_dataset("yuriseki/JParaCrawl", split="train[:100]")
                print(f"Loaded {len(dataset)} examples from JParaCrawl")
            except Exception as e3:
                print(f"Error loading JParaCrawl dataset: {e3}")
                # Last resort, use wmt14 and map from German, but print a warning
                print("WARNING: Could not load any Japanese datasets, falling back to WMT14 with German mapped to Japanese")
                dataset = load_dataset("wmt14", "de-en", split="validation[:100]")
                
                # Map the German entries to Japanese for demonstration purposes
                print("Mapping dataset to support en-ja pair (demo only)...")
                dataset = dataset.map(lambda example: {
                    "translation": {
                        "en": example["translation"]["en"],
                        "ja": example["translation"]["de"]  # Using German as a stand-in for Japanese
                    }
                })
    
    # Build model
    print("Building model...")
    model = build_transformer(config).to(device)
    
    # Getting tokenizer from the model
    tokenizer = model.encoder.tokenizer
    
    # Creating datasets
    train_dataset = TranslationDataset(
        dataset.select(range(80)),  # First 80 examples for training
        tokenizer, 
        src_lang="en", 
        tgt_lang="ja",
        max_length=config["max_seq_len"]
    )
    
    val_dataset = TranslationDataset(
        dataset.select(range(80, 100)),  # Last 20 examples for validation
        tokenizer, 
        src_lang="en", 
        tgt_lang="ja",
        max_length=config["max_seq_len"]
    )
    
    # Creating data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False
    )
    
    # Defining optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.NLLLoss(ignore_index=tokenizer.pad_token_id)
    
    # Training loop
    num_epochs = 2  # using fewer epochs for demo
    train_losses = []
    val_losses = []
     
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # Evaluate
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        
        # Translating example
        if epoch % 1 == 0:
            example = dataset[0]["translation"]["en"]
            translate_example(model, tokenizer, example, "ja")
            
    # Plotting training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()

    # Creating demo directory if it doesn't exist
    demo_dir = config["demo_dir"]
    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)
    
    # Using timestamp for unique filename
    timestamp = get_timestamp()
    plot_path = os.path.join(demo_dir, f"training_history_{timestamp}.png")
    plt.savefig(plot_path)
    print(f"Training history saved to {plot_path}")
    
    # Saving trained model with timestamp (only to demo directory)
    model_path = os.path.join(demo_dir, f"transformer_model_{timestamp}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Demonstrating translation with the trained model
    print("\nTranslation examples:")
    for i in range(5):
        example = dataset[i]["translation"]["en"]
        translate_example(model, tokenizer, example, "ja")

if __name__ == "__main__":
    main() 