"""
This test is for the Encoder class. I created it with the help of Gemini 2.5.
"""

import torch
import math
from typing import List, Tuple, Optional
from model import Transformer, Encoder, Decoder, ProjectionLayer, InputEmbedding, PositionalEncoding # Import all needed

# Define test device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Testing on device: {device} ---")

# --- Test Configuration ---
TEST_CONFIG = {
    "d_model": 128,
    "num_heads": 4,
    "dropout": 0.1,
    "d_ff": 256,
    "tokenizer_name": "bert-base-uncased", # For Encoder's InputEmbedding
    "max_len_pe": 512,
    "target_vocab_size": 10000, # Example target vocab size
    "max_length": None # InputEmbedding padding behavior
}
NUM_ENC_LAYERS = 2 # Fewer layers for faster tests
NUM_DEC_LAYERS = 2
TARGET_PADDING_ID = 0 # Define target padding ID used in dummy data

# --- Helper Function ---
def run_transformer(transformer_model, src_texts, tgt_tokens, return_weights=False):
    """Runs the transformer in eval mode with no gradients."""
    transformer_model.eval()
    with torch.no_grad():
        logits, sa_weights, ca_weights = transformer_model(
            src_texts, tgt_tokens,
            target_padding_id=TARGET_PADDING_ID, # Pass padding ID
            return_last_dec_attn_weights=return_weights
        )
    return logits, sa_weights, ca_weights

# --- Test Functions ---

def test_transformer_forward_shape():
    print("\n--- Test 1: Full Transformer Forward Pass & Shape Check ---")
    # Initialize full model
    encoder = Encoder(config=TEST_CONFIG, n_layers=NUM_ENC_LAYERS, device=device)
    decoder = Decoder(
        target_vocab_size=TEST_CONFIG["target_vocab_size"],
        d_model=TEST_CONFIG["d_model"], n_layers=NUM_DEC_LAYERS,
        num_heads=TEST_CONFIG["num_heads"], d_ff=TEST_CONFIG["d_ff"],
        dropout=TEST_CONFIG["dropout"], max_len_pe=TEST_CONFIG["max_len_pe"], device=device
    )
    projection = ProjectionLayer(TEST_CONFIG["d_model"], TEST_CONFIG["target_vocab_size"], device)
    transformer = Transformer(encoder, decoder, projection).to(device)

    # Dummy data
    test_src = ["Source sentence.", "Another one."]
    test_tgt_len = 10
    test_tgt = torch.randint(1, 500, (len(test_src), test_tgt_len), device=device) # Random target IDs > 0
    test_tgt[:, -2:] = TARGET_PADDING_ID # Add some padding

    logits, _, _ = run_transformer(transformer, test_src, test_tgt)

    print(f"Test 1 Output Logits Shape: {logits.shape}")
    assert logits.shape[0] == len(test_src), f"Expected batch size {len(test_src)}, got {logits.shape[0]}"
    assert logits.shape[1] == test_tgt_len, f"Expected target seq len {test_tgt_len}, got {logits.shape[1]}"
    assert logits.shape[2] == TEST_CONFIG["target_vocab_size"], f"Expected target vocab size {TEST_CONFIG['target_vocab_size']}, got {logits.shape[2]}"
    print("Test 1 Passed: Transformer forward pass completed with correct output dimensions.")

def test_transformer_batch_independence():
    print("\n--- Test 2: Transformer Batch Independence / Determinism ---")
    encoder = Encoder(config=TEST_CONFIG, n_layers=NUM_ENC_LAYERS, device=device)
    decoder = Decoder(TEST_CONFIG["target_vocab_size"], TEST_CONFIG["d_model"], NUM_DEC_LAYERS, TEST_CONFIG["num_heads"], TEST_CONFIG["d_ff"], TEST_CONFIG["dropout"], TEST_CONFIG["max_len_pe"], device)
    projection = ProjectionLayer(TEST_CONFIG["d_model"], TEST_CONFIG["target_vocab_size"], device)
    transformer = Transformer(encoder, decoder, projection).to(device)

    test_src = ["Identical source.", "Identical source."]
    test_tgt = torch.tensor([
        [101, 100, 102, TARGET_PADDING_ID],
        [101, 100, 102, TARGET_PADDING_ID] # Identical target
    ], dtype=torch.long, device=device)

    logits, _, _ = run_transformer(transformer, test_src, test_tgt) # Handles eval() and no_grad()

    logits_seq0 = logits[0]
    logits_seq1 = logits[1]

    are_identical = torch.allclose(logits_seq0, logits_seq1, atol=1e-5) # Use slightly higher tolerance
    print(f"Test 2 Outputs for identical inputs are identical: {are_identical}")
    assert are_identical, "Test 2 Failed: Transformer outputs for identical inputs differ!"
    print("Test 2 Passed: Transformer produces deterministic outputs (in eval mode).")

def test_encoder_padding_in_cross_attn():
    print("\n--- Test 3: Encoder Padding Mask in Cross-Attention ---")
    # Need specific setup to ensure encoder padding
    test_config_pad = TEST_CONFIG.copy()
    # test_config_pad["max_length"] = 15 # Force padding/truncation if needed

    encoder = Encoder(config=test_config_pad, n_layers=NUM_ENC_LAYERS, device=device)
    decoder = Decoder(TEST_CONFIG["target_vocab_size"], TEST_CONFIG["d_model"], NUM_DEC_LAYERS, TEST_CONFIG["num_heads"], TEST_CONFIG["d_ff"], TEST_CONFIG["dropout"], TEST_CONFIG["max_len_pe"], device)
    projection = ProjectionLayer(TEST_CONFIG["d_model"], TEST_CONFIG["target_vocab_size"], device)
    transformer = Transformer(encoder, decoder, projection).to(device)

    test_src = ["Source.", "A much longer source sentence for padding."]
    test_tgt_len = 5
    test_tgt = torch.randint(1, 500, (len(test_src), test_tgt_len), device=device) # Simple target

    # Find padding in encoder input for the first source sentence
    encodings = encoder.tokenizer(test_src, padding='longest', return_tensors='pt').to(device)
    input_ids_src = encodings['input_ids']
    src_padding_token_id = encoder.padding_token_id
    is_padding_in_src0 = (input_ids_src[0] == src_padding_token_id)
    src_padding_indices_seq0 = torch.where(is_padding_in_src0)[0].tolist()
    non_padding_indices_src0 = torch.where(~is_padding_in_src0)[0].tolist()

    print(f"Test 3: Source 0 Input IDs: {input_ids_src[0].tolist()}")
    print(f"Test 3: Padding indices in source 0: {src_padding_indices_seq0}")

    if not non_padding_indices_src0:
        print("Test 3 Warning: Source 0 is all padding. Cannot test.")
        return
    if not src_padding_indices_seq0:
        print("Test 3 Skipped: No padding occurred in source 0.")
        return

    # Run forward requesting weights
    _, _, cross_attn_weights = run_transformer(transformer, test_src, test_tgt, return_weights=True)

    assert cross_attn_weights is not None, "Test 3 Failed: Did not receive cross-attention weights."

    # Check cross-attention weights for the FIRST target token in the FIRST sequence
    # looking at the ENCODER PADDING positions
    # cross_attn_weights shape: (batch, heads, TgtSeqLen, SrcSeqLen)
    query_pos = 0 # First target token
    attn_weights_for_tgt_token = cross_attn_weights[0, :, query_pos, :] # (heads, SrcSeqLen)

    # Get weights assigned to padded source positions
    weights_on_src_padding = attn_weights_for_tgt_token[:, src_padding_indices_seq0] # (heads, num_src_padding)

    # Check sums
    sum_of_weights = attn_weights_for_tgt_token.sum(dim=-1)
    assert torch.allclose(sum_of_weights, torch.ones_like(sum_of_weights), atol=1e-5), \
        f"Test 3 Failed: Cross-Attention weights for target query {query_pos} do not sum to 1."
    print("Test 3: Cross-Attention weights sum to 1.")

    # Check weights on padding
    max_weight_on_padding = torch.max(weights_on_src_padding).item() if weights_on_src_padding.numel() > 0 else 0.0
    print(f"Test 3: Max cross-attention weight on source padding (for target token {query_pos}): {max_weight_on_padding:.2E}")

    padding_ignored = max_weight_on_padding < 1e-6
    assert padding_ignored, f"Test 3 Failed: Significant cross-attention weight ({max_weight_on_padding:.2E}) found on source padding!"
    print("Test 3 Passed: Encoder padding correctly ignored by cross-attention.")


def test_causal_mask_effectiveness():
    print("\n--- Test 4: Causal Mask Effectiveness (Decoder Self-Attention) ---")
    encoder = Encoder(config=TEST_CONFIG, n_layers=NUM_ENC_LAYERS, device=device)
    decoder = Decoder(TEST_CONFIG["target_vocab_size"], TEST_CONFIG["d_model"], NUM_DEC_LAYERS, TEST_CONFIG["num_heads"], TEST_CONFIG["d_ff"], TEST_CONFIG["dropout"], TEST_CONFIG["max_len_pe"], device)
    projection = ProjectionLayer(TEST_CONFIG["d_model"], TEST_CONFIG["target_vocab_size"], device)
    transformer = Transformer(encoder, decoder, projection).to(device)

    test_src = ["A source sentence."] # Only need one source
    test_tgt_len = 6
    test_tgt = torch.randint(1, 500, (len(test_src), test_tgt_len), device=device) # No padding needed here

    # Run forward requesting weights
    _, self_attn_weights, _ = run_transformer(transformer, test_src, test_tgt, return_weights=True)

    assert self_attn_weights is not None, "Test 4 Failed: Did not receive self-attention weights."

    # Check weights for a token attending to FUTURE tokens
    # self_attn_weights shape: (batch, heads, TgtSeqLen, TgtSeqLen)
    # Example: Check weights for query token at position 1 (second token)
    query_pos = 1
    if test_tgt_len <= query_pos:
        print("Test 4 Skipped: Sequence too short to test causal mask effectively.")
        return

    attn_weights_for_token = self_attn_weights[0, :, query_pos, :] # (heads, TgtSeqLen)

    # Indices of future tokens (key positions > query_pos)
    future_indices = list(range(query_pos + 1, test_tgt_len))

    if not future_indices:
        print(f"Test 4 Skipped: No future tokens to check for query position {query_pos}.")
        return

    # Get weights assigned to future key positions
    weights_on_future = attn_weights_for_token[:, future_indices] # (heads, num_future_tokens)

    # Check sums
    sum_of_weights = attn_weights_for_token.sum(dim=-1)
    assert torch.allclose(sum_of_weights, torch.ones_like(sum_of_weights), atol=1e-5), \
        f"Test 4 Failed: Self-Attention weights for target query {query_pos} do not sum to 1."
    print("Test 4: Self-Attention weights sum to 1.")

    # Check weights on future positions
    max_weight_on_future = torch.max(weights_on_future).item() if weights_on_future.numel() > 0 else 0.0
    print(f"Test 4: Max self-attention weight on future tokens (for query token {query_pos}): {max_weight_on_future:.2E}")

    causality_maintained = max_weight_on_future < 1e-6
    assert causality_maintained, f"Test 4 Failed: Significant self-attention weight ({max_weight_on_future:.2E}) found on future tokens!"
    print("Test 4 Passed: Causal mask correctly prevents attention to future tokens.")


def test_transformer_gradient_flow():
    print("\n--- Test 5: Transformer Gradient Flow Check ---")
    grad_config = TEST_CONFIG.copy()
    grad_config["dropout"] = 0.1 # Ensure dropout is non-zero
    encoder = Encoder(config=grad_config, n_layers=NUM_ENC_LAYERS, device=device)
    decoder = Decoder(grad_config["target_vocab_size"], grad_config["d_model"], NUM_DEC_LAYERS, grad_config["num_heads"], grad_config["d_ff"], grad_config["dropout"], grad_config["max_len_pe"], device)
    projection = ProjectionLayer(grad_config["d_model"], grad_config["target_vocab_size"], device)
    transformer = Transformer(encoder, decoder, projection).to(device)
    transformer.train() # Set to training mode

    test_src = ["Test sequence for gradients."]
    test_tgt = torch.randint(1, 500, (len(test_src), 5), device=device) # Dummy target

    # Forward pass (no torch.no_grad())
    logits, _, _ = transformer(test_src, test_tgt)

    # Dummy loss
    loss = logits.mean()
    print(f"Test 5: Calculated dummy loss: {loss.item()}")

    # Backpropagate
    loss.backward()

    # Check gradients exist for parameters in all components
    components = {'encoder': encoder, 'decoder': decoder, 'projection': projection}
    all_grads_ok = True
    for comp_name, component in components.items():
        found_grad_comp = False
        no_grad_params_comp = []
        print(f"  Checking gradients for: {comp_name}")
        for name, param in component.named_parameters():
            if param.requires_grad:
                if param.grad is not None and param.grad.abs().sum() > 0: # Check non-zero grad sum
                    found_grad_comp = True
                elif param.grad is None:
                     no_grad_params_comp.append(name)
                # else: grad is all zeros, might be okay but less informative

        if not found_grad_comp:
            print(f"  Test 5 FAILED for {comp_name}: No non-zero gradients found!")
            all_grads_ok = False
        if no_grad_params_comp:
            print(f"  Test 5 Warning ({comp_name}): No gradients found for parameters: {no_grad_params_comp}")
        # else: # Optional verbose success
            # print(f"  Test 5 OK for {comp_name}: Gradients found.")

    assert all_grads_ok, "Test 5 Failed: Gradients did not flow back correctly to all components."
    print("Test 5 Passed: Gradients flowed back to parameters in Encoder, Decoder, and Projection.")


# --- Run Tests ---
if __name__ == "__main__":
    # Make sure model.py is saved with the modifications for returning weights
    try:
        test_transformer_forward_shape()
        test_transformer_batch_independence()
        test_encoder_padding_in_cross_attn()
        test_causal_mask_effectiveness()
        test_transformer_gradient_flow()
        print("\n--- All Transformer Tests Completed Successfully ---")
    except AssertionError as e:
        print(f"\n--- TEST FAILED ---")
        print(e)
    except Exception as e:
        print(f"\n--- UNEXPECTED ERROR DURING TESTING ---")
        print(e)
        import traceback
        traceback.print_exc()
