import torch
from config import get_config
from simplified_model import build_transformer_from_config

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    # Load configuration
    config = get_config()
    
    # Build model
    print("Building model...")
    model = build_transformer_from_config(config).to(device)
    
    # Get tokenizer from the model
    tokenizer = model.encoder.tokenizer
    
    # Define sample texts to demonstrate with
    sample_texts = [
        "Hello, how are you doing today?",
        "Transformers are powerful neural network architectures.",
        "I'm learning about natural language processing.",
        "This simplified model makes it easier to understand transformers."
    ]
    
    # Generate target sequence for each sample text
    print("\nModel Demonstration (auto-regressive generation):")
    print("=" * 60)
    print("The model will generate sequences from the input text.")
    print("Since it's not trained, outputs won't be meaningful translations.")
    print("This just demonstrates the architecture's functionality.")
    print("=" * 60)
    
    for i, text in enumerate(sample_texts):
        print(f"\nExample {i+1}:")
        print(f"Input text: {text}")
        
        # Tokenize input
        input_tokens = tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)
        
        # Generate sequence
        with torch.no_grad():
            generated_ids = model.generate([text], max_len=20)
        
        # Decode generated sequence
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print(f"Generated sequence: {generated_text[0]}")
        
        # Show attention pattern (just random at this point, but for demonstration)
        print("Attention pattern would be visualized here in a real application")
        
    # Demonstrate the encoder-decoder process
    print("\nEncoder-Decoder Process Demonstration:")
    print("=" * 60)
    
    # Take the first sample
    sample = sample_texts[0]
    print(f"Input: {sample}")
    
    # Create a target sequence (just a placeholder)
    target_seq = torch.ones((1, 5)).long() * tokenizer.cls_token_id
    target_seq = target_seq.to(device)
    
    # Forward pass - use a list for the text input
    print("Running forward pass...")
    with torch.no_grad():
        output = model([sample], target_seq)
    
    print(f"Output shape: {output.shape}")
    print("This represents logits over the vocabulary for each position")
    
    # Show model architecture summary
    print("\nModel Architecture:")
    print("=" * 60)
    print(f"Encoder layers: {config['num_layers']}")
    print(f"Decoder layers: {config['num_layers']}")
    print(f"Attention heads: {config['num_heads']}")
    print(f"Model dimension: {config['d_model']}")
    print(f"Feed-forward dimension: {config['d_ff']}")
    print(f"Vocabulary size: {len(tokenizer)}")

if __name__ == "__main__":
    main() 