from model import Encoder

input_text = "Another day of waking up with the privilege"

TRANSFORMER_CONFIG = {
    "d_model": 512,
    "num_heads": 16,
    "dropout": 0.1,
    "causal_masking": True
}

encoder = Encoder(TRANSFORMER_CONFIG)

output = encoder(input_text)

print(f"\nEncoder's sequence length: {encoder.seq_length}")
print(f"\nEncoder's vocab size: {encoder.vocab_size()}")

print(f"\nEncoder's output shape: {output.shape}")
print(f"\nEncoder's output:\n{output}\n")