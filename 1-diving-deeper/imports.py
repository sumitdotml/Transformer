import torch
import torch.nn as nn
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

vocab_size = tokenizer.n_vocab  # 50257
d_model = 512

embedding_matrix = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)


def tokenize_text(text):
    token_ids = tokenizer.encode(text)
    embeddings = embedding_matrix(torch.stack([torch.tensor(token_ids)], dim=0))
    return token_ids, embeddings


def tokenize_batch(texts):
    token_ids_list = [tokenizer.encode(text) for text in texts]

    # need the max text size out of all the texts
    max_seq_len = max([len(token_ids) for token_ids in token_ids_list])

    # pad all the texts to the max sequence length (for batch processing)
    padded_token_ids = []
    for ids in token_ids_list:
        padded_ids = ids + [0] * (max_seq_len - len(ids))  # Padding with 0s
        padded_token_ids.append(padded_ids)

    # Convert to tensor and get embeddings
    tokens_tensor = torch.tensor(padded_token_ids)
    embeddings = embedding_matrix(tokens_tensor)

    return token_ids_list, embeddings


def decode_text(token_ids):
    return tokenizer.decode(token_ids)


def decode_batch(token_ids_list):
    return [decode_text(token_ids) for token_ids in token_ids_list]


def positional_encoding_original_paper(input_embeddings):
    batch, seq_len, d_model = input_embeddings.shape
    device = input_embeddings.device

    # Create position and dimension indices
    pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    dim = torch.arange(d_model, dtype=torch.float32, device=device)

    # Compute angle rates
    angle_rates = pos / (10000 ** (2 * dim / d_model))

    # Initialize encoding matrix
    pe = torch.zeros(seq_len, d_model, device=device)

    # Apply sine to even indices, cosine to odd indices
    pe[:, 0::2] = torch.sin(angle_rates[:, 0::2])
    pe[:, 1::2] = torch.cos(angle_rates[:, 1::2])

    # Add batch dimension and return
    return pe.unsqueeze(0).expand(batch, -1, -1) + input_embeddings


def positional_encoding(input_embeddings):
    batch, seq_len, d_model = input_embeddings.shape
    pos = torch.arange(seq_len).unsqueeze(1)
    dim = torch.arange(d_model)

    angle_rates = pos * torch.exp((-2 * dim * torch.log(torch.tensor(10000))) / d_model)

    pe = torch.zeros(seq_len, d_model)  # positional encoding initialized
    pe[:, 0::2] = torch.sin(angle_rates[:, 0::2])
    pe[:, 1::2] = torch.cos(angle_rates[:, 1::2])

    # add batch dimension
    pe = pe.unsqueeze(0).expand(batch, -1, -1) + input_embeddings
    return pe
