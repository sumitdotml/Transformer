import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import our simplified transformer
from config import get_config
from simplified_model import build_transformer_from_config

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
        
        # Tokenize source text (the forward method will handle this)
        
        # Tokenize target text (need token ids here for input to decoder)
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
        # Get source and target
        src_texts = batch["src_text"]
        tgt_tokens = batch["tgt_tokens"].to(device)
        
        # Shift targets for teacher forcing 
        # (model should predict next token given previous tokens)
        input_tgt = tgt_tokens[:, :-1]
        output_tgt = tgt_tokens[:, 1:]
        
        # Forward pass
        output = model(src_texts, input_tgt)
        
        # Reshape output and target for loss calculation
        # output: [batch, seq_len, vocab_size] -> [batch*seq_len, vocab_size]
        # target: [batch, seq_len] -> [batch*seq_len]
        output = output.reshape(-1, output.size(-1))
        output_tgt = output_tgt.reshape(-1)
        
        # Calculate loss
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
            # Get source and target
            src_texts = batch["src_text"]
            tgt_tokens = batch["tgt_tokens"].to(device)
            
            # Shift targets for teacher forcing
            input_tgt = tgt_tokens[:, :-1]
            output_tgt = tgt_tokens[:, 1:]
            
            # Forward pass
            output = model(src_texts, input_tgt)
            
            # Reshape output and target for loss calculation
            output = output.reshape(-1, output.size(-1))
            output_tgt = output_tgt.reshape(-1)
            
            # Calculate loss
            loss = criterion(output, output_tgt)
            total_loss += loss.item()
            
    return total_loss / len(data_loader)

def translate_example(model, tokenizer, text, tgt_lang, src_lang="en", max_len=50):
    """Translate a single text example"""
    model.eval()
    
    # Generate with the model
    with torch.no_grad():
        generated_ids = model.generate([text], max_len=max_len)
    
    # Decode generated token IDs to text
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    print(f"{src_lang}: {text}")
    print(f"{tgt_lang}: {generated_text[0]}")
    return generated_text[0]

def main():
    # Load configuration
    config = get_config()
    
    # Load dataset (small subset of WMT14)
    print("Loading dataset...")
    dataset = load_dataset("wmt14", "de-en", split="validation[:100]")
    
    # Build model
    print("Building model...")
    model = build_transformer_from_config(config).to(device)
    
    # Get tokenizer from the model
    tokenizer = model.encoder.tokenizer
    
    # Create datasets
    train_dataset = TranslationDataset(
        dataset.select(range(80)),  # First 80 examples for training
        tokenizer, 
        src_lang="en", 
        tgt_lang="de",
        max_length=config["max_seq_len"]
    )
    
    val_dataset = TranslationDataset(
        dataset.select(range(80, 100)),  # Last 20 examples for validation
        tokenizer, 
        src_lang="en", 
        tgt_lang="de",
        max_length=config["max_seq_len"]
    )
    
    # Create data loaders
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
    
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.NLLLoss(ignore_index=tokenizer.pad_token_id)
    
    # Training loop
    num_epochs = 5  # For demo, use fewer epochs
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
        
        # Translate example
        if epoch % 1 == 0:
            example = dataset[0]["translation"]["en"]
            translate_example(model, tokenizer, example, "de")
            
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.savefig("training_history.png")
    print("Training history saved to training_history.png")
    
    # Save trained model
    torch.save(model.state_dict(), "transformer_model.pth")
    print("Model saved to transformer_model.pth")
    
    # Demonstrate translation with the trained model
    print("\nTranslation examples:")
    for i in range(5):
        example = dataset[i]["translation"]["en"]
        translate_example(model, tokenizer, example, "de")

if __name__ == "__main__":
    main() 