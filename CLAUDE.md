# CLAUDE.md - Transformer Architecture Implementation Guide

## Project Commands
- Run Python script: `python 0-encoding.py`
- Run notebook: `jupyter notebook 1-simplest_selfattn.ipynb` or just use IDEs like VSCode, Cursor, etc.
- Install dependencies: `pip install -r requirements.txt`

## Code Style Guidelines
- **Imports**: torch first, then other libraries (numpy, matplotlib, tiktoken)
- **Naming**: CamelCase for classes, snake_case for variables and functions
- **Matrix naming**: Follow transformer paper conventions (W_q, W_k, W_v)
- **Documentation**: Use markdown cells with LaTeX for mathematical explanations
- **Defaults**: Use smaller embedding dimensions (5) for educational examples
- **Reproducibility**: Set torch.manual_seed(42) for consistent results
- **Error handling**: Use assertions for tensor shape validation
- **Formatting**: Use f-strings for printing tensor shapes and dimensions
- **Comments**: Include detailed explanation of mathematical operations

## Structure
The repository implements transformer architecture components step-by-step (loosely speaking):
1. Tokenization/embedding (0-encoding.py)
2. Self-attention (1-simplest_selfattn.ipynb)
3. Masked self-attention (2-simplest_masked_selfattn.ipynb)