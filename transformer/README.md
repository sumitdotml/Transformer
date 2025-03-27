# Transformer

This directory contains the code for the Transformer model.

## Files

- `model.py`: Contains the Transformer class.
- `test.py`: A basic test file to test whether the model is working as expected.
- `simplified_model.py`: A clean, modular implementation of the transformer architecture that's easier to understand than the original.
- `demo_inference.py`: Shows the basic functionality of the model by generating sequences and exploring the model's output shapes.
- `demo.py`: A more comprehensive demo that sets up a small translation training task using a subset of the WMT14 dataset.
- `visualization_simple.py`: Tools to visualize key components of the transformer, including attention patterns and positional encodings.

The simplified model successfully implements the core transformer architecture with:

- Pre-norm residual connections for better training stability
- Modular components with clear interfaces
- Built-in text handling and generation capabilities
- Comprehensive parameter configuration

When running the visualization script, it generated several visualization files:

- `attention_patterns.png`: Shows the attention patterns across different heads
- `positional_encoding.png`: Displays the positional encoding patterns
- `output_comparison.png`: Compares input and output lengths for various texts

The demonstration shows that the model can:

- Process raw text input directly
- Generate sequences with beam search
- Properly handle masking for attention
- Maintain the correct tensor shapes throughout the network

Since this is an untrained model, the generated sequences aren't meaningful translations - they're just demonstrating the functionality of the architecture itself. With proper training, this model could be used for translation, summarization, or other sequence-to-sequence tasks.

## TODO: Original Paper Replication

- [x] Multi-head attention with encoded embeddings
- [x] Feed forward
- [x] Add & Norm
- [x] Residual connection
- [x] Encoder
- [ ] Decoder
- [ ] Connection between encoder and decoder
- [ ] Create a working architecture with a simple string as input
- [ ] Expand the architecture to work with batches
- [ ] Dataset, DataLoader, etc.
- [ ] Training loop
- [ ] Evaluation metrics
- [ ] Inference

---

<p align="center">
    <a href="#table-of-contents">Go to the top</a> | <a href="../1-diving-deeper/README.md">Go back to 1-diving-deeper</a>
</p>
