from model import Encoder
import torch

input_text = "Another day of waking up with the privilege"

TRANSFORMER_CONFIG = {
    "d_model": 512,
    "num_heads": 16,
    "dropout": 0.1,
    "masking": False,
}

encoder = Encoder(TRANSFORMER_CONFIG)

output = encoder(input_text)

print(
    f"\n======================================= Encoder ======================================="
)
print(f"\nSequence length: {encoder.seq_length}")
print(f"\nVocab size: {encoder.vocab_size()}")
print(f"\nOutput shape: {output.shape}")
print(f"Output:\n{output}\n")

print(
    f"======================================================================================="
)


def test_encoder_sanity():
    config = {"d_model": 512, "num_heads": 8, "dropout": 0.1, "masking": True}
    encoder = Encoder(config)

    # Test gradient flow
    test_input = "Test sequence"
    output = encoder(test_input)
    loss = output.sum()
    loss.backward()

    # Check gradients exist
    for param in encoder.parameters():
        assert param.grad is not None, "No gradient flow!"

    # Test attention patterns
    attn_weights = encoder.layers[0].self_attn_block.attn_weights
    assert torch.allclose(
        attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1)), atol=1e-5
    ), "Attention weights don't sum to 1"


if __name__ == "__main__":
    test_encoder_sanity()
