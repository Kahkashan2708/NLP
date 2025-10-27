# 🧠 Tokenization — NLP Interview & Revision Notes

> **Goal:** Quick reference + interview revision notes on Tokenization (classical + modern NLP)

---

## 🗂️ Table of Contents

1. [📘 Definition](#-1-definition)
2. [🎯 Why Tokenization Matters](#-2-why-tokenization-matters)
3. [📚 Key Concepts & Terms](#-3-key-concepts--terms)
4. [⚙️ Common Tokenizer Types](#-4-common-tokenizer-types)
5. [🧩 Important Functions](#-5-important-functions-hugging-face--general)
6. [🧠 Practical Comparisons](#-6-practical-comparisons)
7. [💻 Example — Hugging Face Tokenizer](#-7-example--hugging-face-tokenizer)
8. [💬 Common Interview Q&A](#-8-common-interview-qa)
9. [⚡ One-Minute Cheat Sheet](#-9-one-minute-cheat-sheet)
10. [📂 References](#-10-references)

---

## 📘 1. Definition

**Tokenization** → Process of splitting raw text into atomic units (“tokens”) that a model can process.

**Goal:**  
Convert **text → tokens → token IDs (integers)** using a consistent vocabulary mapping.

---

## 🎯 2. Why Tokenization Matters

- Controls **vocabulary size** and handling of **rare / OOV words**
- Affects **model input length** and compute cost
- Determines **semantic granularity** (words vs. subwords)
- Impacts downstream tasks (classification, generation, QA)

---

## 📚 3. Key Concepts & Terms

| 🧩 Term | 🔍 Meaning |
|------|----------|
| **Vocabulary** | Mapping token ↔ id |
| **OOV (Out-of-Vocab)** | Words not in vocab, handled via `<UNK>` or subword split |
| **Subword Tokenization** | Splits rare words into smaller known units |
| **BPE (Byte-Pair Encoding)** | Merges frequent symbol pairs iteratively |
| **WordPiece** | Like BPE but uses likelihood-based merges (used in BERT) |
| **Unigram** | Probabilistic subword model (used by SentencePiece) |
| **Character Tokenization** | Each character is a token |
| **Normalization** | Lowercasing, Unicode NFKC, accent stripping |
| **Byte-level Tokenization** | Encodes raw bytes (handles any language) |
| **Special Tokens** | `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]`, `[MASK]` |
| **Offsets** | Map tokens ↔ character spans (important for NER/QA) |
| **Padding / Truncation** | Adjust sequence lengths |
| **Attention Mask** | `1` for real tokens, `0` for pads |
| **Token Type IDs** | Distinguish sentence A vs B (BERT) |
| **Detokenization** | Convert tokens/ids → text |

---

## ⚙️ 4. Common Tokenizer Types

| ⚙️ Type | 📖 Description | 🧠 Examples |
|------|--------------|-----------|
| **Whitespace / Rule-based** | Splits by spaces/punctuation | Simple preprocessing |
| **Regex / NLTK / spaCy** | Linguistic tokenization | POS tagging, parsing |
| **WordPiece** | Subword merges by likelihood | BERT |
| **BPE** | Subword merges by frequency | GPT, RoBERTa |
| **SentencePiece** | Language-agnostic subword model | T5, ALBERT |
| **Byte-level BPE** | Operates on bytes | GPT-2, GPT-3 |

---

## 🧩 5. Important Functions (Hugging Face & General)

| 🧮 Function | 🧰 Purpose |
|-----------|----------|
| `tokenize(text)` | Split text → list of tokens |
| `encode(text, add_special_tokens=True)` | Text → token IDs |
| `batch_encode_plus(texts)` | Batch encode with padding & masks |
| `decode(ids, skip_special_tokens=True)` | Token IDs → text |
| `convert_tokens_to_ids(tokens)` | Map tokens → IDs |
| `convert_ids_to_tokens(ids)` | Map IDs → tokens |
| `add_special_tokens(tokens)` | Add `[CLS]`, `[SEP]`, etc. |
| `pad_sequences(sequences, maxlen)` | Pad sequences |
| `get_attention_mask(input_ids)` | Mask: 1 for tokens, 0 for pads |
| `encode_plus(..., return_offsets_mapping=True)` | Get token-char alignment |
| `train_tokenizer(corpus, vocab_size, model_type)` | Train new tokenizer |

---

## 🧠 6. Practical Comparisons

| 🔍 Type | ✅ Pros | ⚠️ Cons |
|-------|------|------|
| **Word-level** | Intuitive | OOV problem |
| **Subword-level** | Handles rare words, balanced vocab | More complex |
| **Char-level** | No OOV | Very long sequences |
| **BPE vs WordPiece** | BPE = frequency merges; WordPiece = likelihood merges |
| **SentencePiece** | Trains directly on raw text | Slightly slower |

---

## 💻 7. Example — Hugging Face Tokenizer

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

out = tokenizer(
    "Hello world!",
    padding="max_length",
    max_length=8,
    truncation=True
)

print(out)
# Output:
# {'input_ids': [...], 'attention_mask': [...], 'token_type_ids': [...]}

```

🗣️ **Explain:**

- **`input_ids`** → Token IDs  
- **`attention_mask`** → `1` for real tokens, `0` for pads

| Key | Description |
|-----|--------------|
| `input_ids` | Token IDs — numerical representation of tokens |
| `attention_mask` | `1` for real tokens, `0` for padding tokens |


---

## 💬 8. Common Interview Q&A

| 💡 Question | 🧭 One-line Answer |
|-------------|-------------------|
| Why subword tokenization? | Reduces OOV, keeps manageable vocab, handles rare words |
| What is an attention mask? | Binary mask to ignore padded tokens in attention |
| How to handle OOV at inference? | Use `<UNK>` or subword split; byte-level if multilingual |
| How to map model output to text? | Use `offset_mapping` for token-char span alignment |
| Difference between BPE and WordPiece? | BPE uses frequency merges; WordPiece uses likelihood merges |
| What is `[CLS]` and `[SEP]` token used for? | `[CLS]`: classification embedding, `[SEP]`: sentence separation |
| What’s the tokenizer output format in Hugging Face? | Dict with `input_ids`, `attention_mask`, `token_type_ids` |
| Why SentencePiece doesn’t need pre-tokenization? | It directly learns on raw text (no whitespace split) |

---

## ⚡ 9. One-Minute Cheat Sheet

🚀 **Pipeline:** `text → tokens → IDs → model`  
🧾 **Tokenizer Outputs:** `input_ids`, `attention_mask`, (`token_type_ids`)  
🏷️ **Special Tokens:** `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]`, `[MASK]`  
🧩 **Subword Methods:** `BPE`, `WordPiece`, `Unigram`  
⚙️ **Core Functions:** `tokenize`, `encode`, `decode`, `convert_*`, `batch_encode_plus`  
📁 **Files:** `tokenizer.json`, `vocab.txt`, `merges.txt`  
🧠 **Interview Tip:** Be able to explain “how tokenization affects model performance.”

---

## 📂 10. References

- 📘 [Hugging Face Tokenizers Docs](https://huggingface.co/docs/tokenizers)
- 📄 [BERT: WordPiece Tokenizer Paper (2018)](https://arxiv.org/abs/1810.04805)
- 📄 [SentencePiece Paper (Google, 2018)](https://arxiv.org/abs/1808.06226)
- 📗 [Jurafsky & Martin — *Speech and Language Processing (3rd Ed.)*](https://web.stanford.edu/~jurafsky/slp3/)
- 🎥 [CampusX NLP Series (YouTube)](https://www.youtube.com/playlist?list=PLKnIA16_Rmvb1RYR-iTA_hzckhdONtSW4)
- 📘 [The Illustrated WordPiece & BPE (Blog)](https://huggingface.co/blog/how-tokenizers-work)

---
