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

print(f"\n======================================= Encoder =======================================")
print(f"\nSequence length: {encoder.seq_length}")
print(f"\nVocab size: {encoder.vocab_size()}")
print(f"\nOutput shape: {output.shape}")
print(f"Output:\n{output}\n")

print(f"=======================================================================================")