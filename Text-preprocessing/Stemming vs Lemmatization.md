# Stemming vs Lemmatization (NLTK)

## Definitions
- **Stemming:** NLP process that cuts off affixes to reduce a word to its root or stem form:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}. Stems may not be valid words (e.g. “running” → “runn”:contentReference[oaicite:2]{index=2}).
- **Lemmatization:** NLP process that reduces words to their lemma (dictionary form):contentReference[oaicite:3]{index=3}. It uses morphological analysis with part-of-speech (POS) context to ensure the output is a valid word (e.g. “running” → “run”):contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}.

## Key Differences

| Aspect    | Stemming                                        | Lemmatization                                  |
|-----------|-------------------------------------------------|-----------------------------------------------|
| Approach  | Rule-based affix stripping (e.g. Porter, Lancaster):contentReference[oaicite:6]{index=6} | Dictionary/vocabulary + POS tagging:contentReference[oaicite:7]{index=7}:contentReference[oaicite:8]{index=8} |
| Output    | Root/stem (may be non-word, e.g. “running”→“runn”):contentReference[oaicite:9]{index=9} | Lemma (valid word, e.g. “running”→“run”):contentReference[oaicite:10]{index=10} |
| Speed     | Fast (simple rules, no POS lookup):contentReference[oaicite:11]{index=11} | Slower (needs lexicon lookups and POS):contentReference[oaicite:12]{index=12} |
| Accuracy  | Lower (can over-stem and conflate words):contentReference[oaicite:13]{index=13} | Higher (returns correct base forms):contentReference[oaicite:14]{index=14} |
| Use-case  | Good for quick text processing/search indexing:contentReference[oaicite:15]{index=15} | Good for tasks needing precise meaning (analysis, NLP):contentReference[oaicite:16]{index=16} |

## When to Use (Pros & Cons)
- **Stemming:** Simple and fast (rule-based) – useful when speed is more important than accuracy (e.g. search indexing):contentReference[oaicite:17]{index=17}:contentReference[oaicite:18]{index=18}. *Pros:* reduces data dimensionality quickly. *Cons:* may produce non-words or incorrect stems:contentReference[oaicite:19]{index=19}, losing meaning.
- **Lemmatization:** More accurate (dictionary/POS-driven) – useful when word meaning matters (e.g. sentiment analysis, NLP pipelines):contentReference[oaicite:20]{index=20}:contentReference[oaicite:21]{index=21}. *Pros:* yields real words (handles irregulars, e.g. “better”→“good”:contentReference[oaicite:22]{index=22}). *Cons:* slower and resource-intensive, requires POS tagging:contentReference[oaicite:23]{index=23}.

## NLTK Code Examples
- **PorterStemmer:** 
  ```python
  from nltk.stem import PorterStemmer
  stemmer = PorterStemmer()
  print(stemmer.stem("running"))  # run
