# BertJapanese

Taken from the [BertJapanese](https://huggingface.co/docs/transformers/model_doc/bert-japanese) documentation on Hugging Face.

## Overview

The BERT models trained on Japanese text.

There are models with two different tokenization methods:

- Tokenize with MeCab and WordPiece. This requires some extra dependencies, fugashi which is a wrapper around MeCab.
- Tokenize into characters.

To use MecabTokenizer, you should pip install transformers["ja"] (or pip install -e .["ja"] if you install from source) to install dependencies.

See details on [cl-tohoku repository](https://github.com/cl-tohoku/bert-japanese).

Example of using a model with MeCab and WordPiece tokenization:

```python
import torch
from transformers import AutoModel, AutoTokenizer

bertjapanese = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese")
tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

## Input Japanese Text
line = "吾輩は猫である。"

inputs = tokenizer(line, return_tensors="pt")

print(tokenizer.decode(inputs["input_ids"][0]))
[CLS] 吾輩 は 猫 で ある 。 [SEP]

outputs = bertjapanese(**inputs)
```

Example of using a model with Character tokenization:

```python
bertjapanese = AutoModel.from_pretrained("cl-tohoku/bert-base-japanese-char")
tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-char")

## Input Japanese Text
line = "吾輩は猫である。"

inputs = tokenizer(line, return_tensors="pt")

print(tokenizer.decode(inputs["input_ids"][0]))
[CLS] 吾 輩 は 猫 で あ る 。 [SEP]

outputs = bertjapanese(**inputs)
```

This model was contributed by cl-tohoku.

This implementation is the same as BERT, except for tokenization method. Refer to BERT documentation for API reference information.

## BertJapaneseTokenizer

```python
class transformers.BertJapaneseTokenizer
( vocab_filespm_file = Nonedo_lower_case = Falsedo_word_tokenize = Truedo_subword_tokenize = Trueword_tokenizer_type = 'basic'subword_tokenizer_type = 'wordpiece'never_split = Noneunk_token = '[UNK]'sep_token = '[SEP]'pad_token = '[PAD]'cls_token = '[CLS]'mask_token = '[MASK]'mecab_kwargs = Nonesudachi_kwargs = Nonejumanpp_kwargs = None**kwargs )
```

### Parameters

- vocab_file (str) — Path to a one-wordpiece-per-line vocabulary file.
- spm_file (str, optional) — Path to SentencePiece file (generally has a .spm or .model extension) that contains the vocabulary.
- do_lower_case (bool, optional, defaults to True) — Whether to lower case the input. Only has an effect when do_basic_tokenize=True.
- do_word_tokenize (bool, optional, defaults to True) — Whether to do word tokenization.
- do_subword_tokenize (bool, optional, defaults to True) — Whether to do subword tokenization.
- word_tokenizer_type (str, optional, defaults to "basic") — Type of word tokenizer. Choose from [“basic”, “mecab”, “sudachi”, “jumanpp”].
- subword_tokenizer_type (str, optional, defaults to "wordpiece") — Type of subword tokenizer. Choose from [“wordpiece”, “character”, “sentencepiece”,].
- mecab_kwargs (dict, optional) — Dictionary passed to the MecabTokenizer constructor.
- sudachi_kwargs (dict, optional) — Dictionary passed to the SudachiTokenizer constructor.
- jumanpp_kwargs (dict, optional) — Dictionary passed to the JumanppTokenizer constructor.

Construct a BERT tokenizer for Japanese text.

This tokenizer inherits from PreTrainedTokenizer which contains most of the main methods. Users should refer to: this superclass for more information regarding those methods.

```python
build_inputs_with_special_tokens
( token_ids_0: typing.List[int]token_ids_1: typing.Optional[typing.List[int]] = None ) → List[int]

Parameters

token_ids_0 (List[int]) — List of IDs to which the special tokens will be added.
token_ids_1 (List[int], optional) — Optional second list of IDs for sequence pairs.
Returns

List[int]

List of input IDs with the appropriate special tokens.
```

Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and adding special tokens. A BERT sequence has the following format:

single sequence: [CLS] X [SEP]
pair of sequences: [CLS] A [SEP] B [SEP]

```python
convert_tokens_to_string
( tokens )

Converts a sequence of tokens (string) in a single string.

create_token_type_ids_from_sequences
( token_ids_0: typing.List[int]token_ids_1: typing.Optional[typing.List[int]] = None ) → List[int]

Parameters

token_ids_0 (List[int]) — List of IDs.
token_ids_1 (List[int], optional) — Optional second list of IDs for sequence pairs.
Returns

List[int]

List of token type IDs according to the given sequence(s).
```

Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence

pair mask has the following format:

0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
| first sequence | second sequence |
If token_ids_1 is None, this method only returns the first portion of the mask (0s).

```python
get_special_tokens_mask
( token_ids_0: typing.List[int]token_ids_1: typing.Optional[typing.List[int]] = Nonealready_has_special_tokens: bool = False ) → List[int]

Parameters

token_ids_0 (List[int]) — List of IDs.
token_ids_1 (List[int], optional) — Optional second list of IDs for sequence pairs.
already_has_special_tokens (bool, optional, defaults to False) — Whether or not the token list is already formatted with special tokens for the model.
Returns

List[int]

A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
```

Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding special tokens using the tokenizer prepare_for_model method.
