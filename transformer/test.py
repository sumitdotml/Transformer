"""
This test is for the Encoder class. I created it with the help of Gemini 2.5.
"""

import torch
from model import Encoder

# Define test device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Testing on device: {device} ---")

# --- Test Configuration ---
# Use a smaller model for faster testing if desired
TEST_CONFIG = {
    "d_model": 512,  # Smaller embedding dimension
    "num_heads": 8,  # Fewer heads (ensure d_model % num_heads == 0)
    "dropout": 0.1,  # Dropout doesn't matter much in eval mode, but setting it
    "d_ff": 2048,  # Smaller feed-forward dimension
    "tokenizer_name": "bert-base-uncased",  # check for more here: https://huggingface.co/models
    "max_len_pe": 512,  # Max length for Positional Encoding matrix
    "max_length": None,  # Pad to longest in batch for tests unless specified otherwise
}
NUM_TEST_LAYERS = 6  # layers in the encoder


# --- Helper Function (Optional, for cleaner tests) ---
def run_encoder(encoder_model, texts, return_weights=False):
    """Runs the encoder in eval mode with no gradients."""
    encoder_model.eval()  # Set to evaluation mode
    with torch.no_grad():  # Disable gradient calculations
        output, weights = encoder_model(
            texts, return_last_layer_attn_weights=return_weights
        )
    return output, weights


# --- Test Functions ---


def test_forward_pass_shape():
    print("\n--- Test 1: Basic Forward Pass & Shape Check ---")
    encoder = Encoder(config=TEST_CONFIG, n_layers=NUM_TEST_LAYERS, device=device)
    test_texts = [
        "This is the first sentence.",
        "Here is another one, slightly longer.",
        "A short sentence.",
        "The correct person to do this job is William, and I'll tell you why.",
    ]
    # --- CORRECTED LINE ---
    # Get the tokenizer object first, then call it
    tokenizer_obj = encoder.tokenizer
    encodings = tokenizer_obj(test_texts, padding="longest", return_tensors="pt")
    # --- END CORRECTION ---
    expected_seq_len = encodings["input_ids"].shape[1]

    encoder_output, _ = run_encoder(encoder, test_texts)

    print(f"Test 1 Output Shape: {encoder_output.shape}")
    assert encoder_output.shape[0] == len(
        test_texts
    ), f"Expected batch size {len(test_texts)}, got {encoder_output.shape[0]}"
    assert (
        encoder_output.shape[1] == expected_seq_len
    ), f"Expected seq len {expected_seq_len}, got {encoder_output.shape[1]}"
    assert (
        encoder_output.shape[2] == TEST_CONFIG["d_model"]
    ), f"Expected d_model {TEST_CONFIG['d_model']}, got {encoder_output.shape[2]}"
    print("Test 1 Passed: Basic forward pass completed with correct output dimensions.")


def test_batch_independence():
    print("\n--- Test 2: Batch Independence / Determinism ---")
    encoder = Encoder(config=TEST_CONFIG, n_layers=NUM_TEST_LAYERS, device=device)
    # IMPORTANT: Set to eval mode to disable dropout for this test
    encoder.eval()
    test_texts = [
        "This is a test sentence.",
        "This is a test sentence.",  # Identical sentence
    ]
    encoder_output, _ = run_encoder(
        encoder, test_texts
    )  # run_encoder handles eval() and no_grad()

    output_seq0 = encoder_output[0]  # Output for the first instance
    output_seq1 = encoder_output[1]  # Output for the second instance

    are_identical = torch.allclose(output_seq0, output_seq1, atol=1e-6)
    print(f"Test 2 Outputs for identical inputs are identical: {are_identical}")
    assert (
        are_identical
    ), "Test 2 Failed: Outputs for identical inputs in a batch differ!"
    print(
        "Test 2 Passed: Model produces deterministic outputs for identical inputs in a batch (in eval mode)."
    )


def test_padding_mask_effectiveness():
    print("\n--- Test 3: Padding Mask Effectiveness ---")
    encoder = Encoder(config=TEST_CONFIG, n_layers=NUM_TEST_LAYERS, device=device)
    test_texts = [
        "Short sentence.",  # Will be padded
        "This is a much longer sentence to ensure padding happens.",
    ]

    # --- CORRECTED LINES ---
    # Get the tokenizer object first, then call it
    tokenizer_obj = encoder.tokenizer
    encodings = tokenizer_obj(test_texts, padding="longest", return_tensors="pt").to(
        device
    )
    # --- END CORRECTION ---
    input_ids = encodings["input_ids"]
    # Access padding_token_id via property
    padding_token_id = encoder.padding_token_id
    is_padding_in_seq0 = input_ids[0] == padding_token_id
    padding_indices_seq0 = torch.where(is_padding_in_seq0)[0].tolist()
    non_padding_indices_seq0 = torch.where(~is_padding_in_seq0)[0].tolist()

    print(f"Test 3: Sequence 0 Input IDs: {input_ids[0].tolist()}")
    print(f"Test 3: Padding indices in sequence 0: {padding_indices_seq0}")

    if not non_padding_indices_seq0:
        print(
            "Test 3 Warning: Sequence 0 consists only of padding/special tokens. Cannot test attention."
        )
        return
    if not padding_indices_seq0:
        print("Test 3 Skipped: No padding occurred in sequence 0 for this batch.")
        return

    first_non_padding_idx_seq0 = non_padding_indices_seq0[
        0
    ]  # Index of first real token
    print(
        f"Test 3: First non-padding index in sequence 0: {first_non_padding_idx_seq0}"
    )

    # Run forward pass requesting attention weights from the last layer
    _, last_attn_weights = run_encoder(encoder, test_texts, return_weights=True)

    assert (
        last_attn_weights is not None
    ), "Test 3 Failed: Did not receive attention weights."

    # Check attention weights for the FIRST non-padding token in the FIRST sequence
    query_token_idx = first_non_padding_idx_seq0
    attn_weights_for_token = last_attn_weights[
        0, :, query_token_idx, :
    ]  # (num_heads, seq_len_k)

    # Get the weights specifically for the padded key positions
    weights_on_padding = attn_weights_for_token[
        :, padding_indices_seq0
    ]  # (num_heads, num_padding_tokens)

    # Check if attention weights sum to 1 (basic check for softmax)
    sum_of_weights = attn_weights_for_token.sum(dim=-1)  # Sum across keys for each head
    assert torch.allclose(
        sum_of_weights, torch.ones_like(sum_of_weights), atol=1e-5
    ), f"Test 3 Failed: Attention weights for query {query_token_idx} do not sum to 1."
    print("Test 3: Attention weights sum to 1 (checked for one token).")

    # Check if all weights on padding are close to zero
    max_weight_on_padding = (
        torch.max(weights_on_padding).item() if weights_on_padding.numel() > 0 else 0.0
    )
    print(
        f"Test 3: Max attention weight on padding tokens (for query token {query_token_idx}): {max_weight_on_padding:.2E}"
    )  # Scientific notation

    padding_ignored = max_weight_on_padding < 1e-6
    assert (
        padding_ignored
    ), f"Test 3 Failed: Significant attention weight ({max_weight_on_padding:.2E}) found on padding tokens!"
    print("Test 3 Passed: Padding tokens correctly ignored by attention.")


def test_positional_encoding_effect():
    print("\n--- Test 4: Positional Encoding Effect (Conceptual Check) ---")
    encoder = Encoder(config=TEST_CONFIG, n_layers=NUM_TEST_LAYERS, device=device)
    test_texts = ["A test sentence long enough for comparison."]
    encoder_output, _ = run_encoder(encoder, test_texts)

    if encoder_output.shape[1] > 1:  # If sequence length > 1
        output_token0 = encoder_output[0, 0, :]  # First token, first sequence
        output_token1 = encoder_output[0, 1, :]  # Second token, first sequence
        are_different = not torch.allclose(output_token0, output_token1, atol=1e-6)
        print(
            f"Test 4: Output for token 0 and token 1 in the same sequence are different: {are_different}"
        )
        assert (
            are_different
        ), "Test 4 Failed: Output for adjacent tokens is identical. Check PE or attention."
        print("Test 4 Passed: Adjacent tokens have different representations.")
    else:
        print("Test 4 Skipped: Sequence length is 1.")


def test_gradient_flow():
    print("\n--- Test 5: Gradient Flow Check ---")
    # Use config with dropout > 0 to ensure dropout layers are active
    grad_config = TEST_CONFIG.copy()
    grad_config["dropout"] = 0.1
    encoder = Encoder(config=grad_config, n_layers=NUM_TEST_LAYERS, device=device)
    # Ensure model is in training mode for dropout and gradient calculation
    encoder.train()

    test_texts = ["Test sequence for gradients."]
    # No torch.no_grad() here
    encoder_output, _ = encoder(test_texts)

    # Simple dummy loss
    loss = encoder_output.mean()  # Use mean instead of sum for stability
    print(f"Test 5: Calculated dummy loss: {loss.item()}")

    # Backpropagate
    loss.backward()

    # Check if gradients exist for parameters
    found_grad = False
    no_grad_params = []
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                found_grad = True
                # Optional: Check magnitude, but just checking existence is often enough
                # print(f"  Grad found for: {name}, Max value: {param.grad.abs().max().item()}")
            else:
                no_grad_params.append(name)

    if not found_grad:
        raise AssertionError("Test 5 Failed: No gradients found for any parameter!")
    if no_grad_params:
        print(f"Test 5 Warning: No gradients found for parameters: {no_grad_params}")
        # This might be okay if e.g. parts of the model were frozen, but unexpected here.
    else:
        print("Test 5 Passed: Gradients flowed back to parameters.")


# --- Run Tests ---
if __name__ == "__main__":
    test_forward_pass_shape()
    test_batch_independence()
    test_padding_mask_effectiveness()
    test_positional_encoding_effect()
    test_gradient_flow()
    print("\n--- All Tests Completed ---")
