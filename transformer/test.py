from model import Transformer
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

input_text = "Another day of waking up with the privilege"

TRANSFORMER_CONFIG = {
    "tokenizer": tokenizer,
    "vocab_size": tokenizer.n_vocab, # in case of tiktoken! could be different for other tokenizers
    "d_model": 512,
    "num_heads": 8,
    "dropout": 0.1,
    "causal_masking": True
}

transformer = Transformer(TRANSFORMER_CONFIG)

output = transformer(input_text)

print(f"\nTransformer's sequence length: {transformer.seq_length}")
print(f"\nTransformer's vocab size: {transformer.vocab_size()}")

print(f"\nTransformer's output shape: {output.shape}")
print(f"\nTransformer's output:\n{output}\n")