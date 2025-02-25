import tiktoken
import torch
import torch.nn as nn

input_text = "This is a pen."
tokenizer = tiktoken.get_encoding("gpt2")
token_ids = tokenizer.encode(input_text)
print(f"Token IDs: {token_ids}")

token_embeddings = torch.randn(len(token_ids), 5)
"""
token_embedding is creating a matrix of (len x 5) size,
with 5 being the size of the embedding dimensions for
simplicity. Ususally we have 512 or 768 or even 1024.
"""

print(f"token_embeddings.shape: {token_embeddings.shape}")
print(f"token_embeddings[0]: {token_embeddings[0]}")
print(f"token_embeddings[0].shape: {token_embeddings[0].shape}")
print(
    f"token_embeddings[0].unsqueeze(0).shape: {
        token_embeddings[0].unsqueeze(0).shape}"
)
