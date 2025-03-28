# Japanese Translation Support

This document outlines the Japanese translation capabilities in our transformer model and how to configure them for optimal performance.

## Tokenization Options

For Japanese translation, the system provides several tokenizer options in order of preference:

1. **Character-based Tokenizer** (`cl-tohoku/bert-base-japanese-char`):

   - Splits Japanese text into individual characters
   - Default for Japanese translation
   - Requires the following dependencies:
     ```bash
     uv pip install protobuf fugashi ipadic unidic-lite
     ```
   - Example: `吾輩は猫である。` → `['吾', '輩', 'は', '猫', 'で', 'あ', 'る', '。']`

2. **MeCab-based Tokenizer** (`cl-tohoku/bert-base-japanese`):

   - Uses MeCab for word segmentation followed by WordPiece
   - Provides more semantically meaningful tokens
   - Requires the same dependencies as the character-based tokenizer
   - Example: `吾輩は猫である。` → `['吾', '##輩', 'は', '猫', 'で', 'ある', '。']`

3. **Universal Byte-level Tokenizer** (`google/byt5-small`):
   - Used as a last resort if the Japanese-specific tokenizers fail
   - Works with any language but not optimal for Japanese
   - Tokenizes at the byte level
   - Example: `吾輩は猫である。` → [byte-level tokens]

## Configuration

Our model is configured to automatically try the Japanese tokenizers in the following order:

1. First tries `cl-tohoku/bert-base-japanese-char` (character-based)
2. If that fails, tries `cl-tohoku/bert-base-japanese` (word-based)
3. If both fail, falls back to the configured tokenizers in the config file
4. As a last resort, uses a universal tokenizer that works with any language

## Required Dependencies

To ensure proper Japanese tokenization, the following dependencies must be installed:

```bash
uv pip install protobuf fugashi ipadic unidic-lite
```

These are included in the `requirements.txt` file, but you may need to install them separately if you encounter tokenization issues.

### Why are these dependencies needed?

- **protobuf**: Required by the Hugging Face tokenizers library
- **fugashi**: Python wrapper for MeCab (Japanese morphological analyzer)
- **ipadic**: Japanese IPA dictionary for MeCab
- **unidic-lite**: A smaller version of UniDic, a Japanese dictionary for MeCab

## Training for Japanese Translation

The default configuration for English to Japanese translation includes:

```python
"en-ja": {
    "num_epochs": 20,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "dropout": 0.2,
    "patience": 5,
    "tokenizer_name": "cl-tohoku/bert-base-japanese-char",
    "fallback_tokenizer_name": "cl-tohoku/bert-base-japanese",
    "second_fallback_tokenizer_name": "google/byt5-small"
}
```

These parameters are optimized for the Japanese language's characteristics, including:

- Higher dropout to prevent overfitting due to the complexity of Japanese characters
- Smaller batch size to accommodate longer sequences
- More training epochs to learn the different writing systems

## Verifying the Tokenizer

You can verify that the Japanese tokenizer is working correctly with the following code:

```python
from transformers import AutoTokenizer

# Test the character-based tokenizer
tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-char')
example = '吾輩は猫である。'
tokens = tokenizer.tokenize(example)
print(f'Tokenization of "{example}": {tokens}')
# Should output: ['吾', '輩', 'は', '猫', 'で', 'あ', 'る', '。']

# Test the word-based tokenizer
tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
tokens = tokenizer.tokenize(example)
print(f'Tokenization of "{example}": {tokens}')
# Should output: ['吾', '##輩', 'は', '猫', 'で', 'ある', '。']
```
