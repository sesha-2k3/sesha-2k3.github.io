---
title: "RAG from Scratch, Part 2: Embeddings, Tokens, and Metadata"
pubDatetime: 2026-10-21T00:00:00Z
featured: true
draft: false
kind: notes
category: AI
tags:
  - rag-series
  - rag
  - llm
description: "Why your user's question can silently vanish before retrieval, why chunk sizes belong in tokens, and what metadata does that embeddings can't."
---

**RAG from Scratch series:** [Part 0: Overview](/posts/rag-00-overview/) · [Part 1: Chunking](/posts/rag-01-chunking/) · **Part 2: Embeddings, tokens, and metadata** · Part 3: The query-time flow (coming soon)

## The question that disappeared

A user pastes a long email into your RAG chatbot and adds one line at the end:

```
Hi team, thanks for the update on Q3 budgets. The finance review went well
and spending is mostly on track, although travel costs rose slightly. We also
discussed hiring plans for next year. Based on this, will the Atlas launch date slip?
```

Here's what the embedding model actually sees, with a toy limit of 32 tokens:

```
User message: 49 tokens, limit 32
What actually gets embedded:
  Hi team , thanks for the update on Q3 budgets . The finance review went well
  and spending is mostly on track , although travel costs rose slightly . We also discussed
Question survived truncation? False
```

The question was cut off. The query vector now means "the opening of an email about budgets," so retrieval returns budget documents instead of the Atlas roadmap. No error is raised, and nothing is logged.

To make it harder to debug, the LLM still receives the full message, because its context window is much larger than the embedding model's limit. So the answer looks like it's ignoring the right documents, when in fact retrieval never searched for them.

This post covers why that happens and how to prevent it, along with two other things you need to get right about embeddings: using one model consistently, and storing metadata.

## Every input becomes the same size

An embedding model always outputs the same number of dimensions, whatever the input length. With `all-MiniLM-L6-v2`, both "Atlas status?" and a 300-word chunk become exactly 384 numbers. The length of the text changes where the vector points, not its shape.

That's why there's no warning when something goes wrong. A truncated input still produces a perfectly normal-looking 384-number vector.

## Use the same model for chunks and queries

Each embedding model learns its own coordinate system, so vectors from different models can't be compared. Mixing them breaks in one of two ways:

- **Different dimensions**, for example MiniLM's 384 versus `text-embedding-3-small`'s 1,536: the vector database rejects the search with a dimension-mismatch error. This is the good case, because you find out.
- **Same dimensions, different models:** no error at all, and the results are meaningless.

This also means switching embedding models requires re-embedding your entire corpus. Two habits make that painless: store the raw text of every chunk, and record the model name in each chunk's metadata. Then a migration is just a batch job.

## Models read tokens, not characters

Models don't see characters. They see **tokens**, which are sub-word pieces of text. English prose averages roughly 4 characters per token, but code, URLs, numbers, and non-English text are often 2–3 characters per token or worse.

Every model has a maximum number of tokens it will read. For `all-MiniLM-L6-v2`, it's 256 (you can check with `model.max_seq_length`). Anything past that is **silently truncated**. You get a vector that represents only the first 256 tokens, and two texts that share their first 256 tokens produce identical vectors.

Here's what that does to chunking. In Part 1, we measured chunk size in characters. Suppose you chunk at 2,000 characters, roughly 500 tokens. About half of every chunk is never embedded. That text is still stored, and it's still sent to the LLM if the chunk is retrieved, but retrieval can only find the chunk through its first half.

### Truncation and token chunking from scratch

The scratch version uses a simple tokenizer that splits on words and punctuation. Real models use sub-word tokenizers, but this is enough to show the effect:

```python
import re

MAX_TOKENS = 32  # stand-in for a real limit such as 256

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text)

def truncate(text, max_tokens=MAX_TOKENS):
    """What an embedding model does silently: keep the first max_tokens."""
    return tokenize(text)[:max_tokens]
```

The fix is to measure chunks in tokens instead of characters. It's the same sliding window as Part 1's fixed-size chunker, applied to a token list:

```python
def token_chunks(text, chunk_tokens, overlap_tokens):
    tokens = tokenize(text)
    step = chunk_tokens - overlap_tokens
    return [" ".join(tokens[i:i + chunk_tokens])
            for i in range(0, max(len(tokens) - overlap_tokens, 1), step)]
```

On the email with 12-token chunks and 3 tokens of overlap:

```
1: Hi team , thanks for the update on Q3 budgets . The
2: budgets . The finance review went well and spending is mostly on
3: is mostly on track , although travel costs rose slightly . We
4: slightly . We also discussed hiring plans for next year .
```

Every chunk is now guaranteed to fit within the limit, and the overlap repeats the last 3 tokens of each chunk at the start of the next.

### The same thing with real tokenizers

In practice, count tokens with the embedding model's own tokenizer:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
print(model.max_seq_length)  # 256

text = "Error code E-4012 means the payment token expired."
n_tokens = len(model.tokenizer(text)["input_ids"])
```

The library script also demonstrates silent truncation directly. It embeds two long texts that share 60 repetitions of the same filler sentence but end differently, one about Atlas and one about bananas. Their cosine similarity comes out at roughly 1.0, because the different endings fall past the limit and are dropped.

For chunking, LangChain has splitters that count tokens with the right tokenizer:

```python
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

# For sentence-transformers models
splitter = SentenceTransformersTokenTextSplitter(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    tokens_per_chunk=200,
    chunk_overlap=30,
)

# For OpenAI-style models, which use tiktoken
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=200, chunk_overlap=30
)
```

Setting `tokens_per_chunk=200` leaves headroom under MiniLM's 256-token limit.

Providers handle the limit differently, which is worth knowing. Local sentence-transformers models truncate silently. OpenAI's embedding API, with a limit of around 8,000 tokens, returns an error instead.

## Fixing truncation on the query side

Short questions never hit the limit, but pasted content does. Even when a pasted message fits under the limit, the pasted text dominates the query vector and drowns out the question. There are four fixes, from best to most basic:

1. **Rewrite the query.** Ask an LLM to turn the user's message into a short search phrase, such as "Project Atlas launch date timeline risks." Embed that phrase for retrieval, then send the original message plus the retrieved chunks to the LLM for the answer.
2. **Separate the question from the pasted content** in your interface, so they can be handled differently.
3. **Chunk long queries.** Search with each piece and merge the results.
4. **At minimum, detect it.** Count tokens and log a warning when a query goes over the limit, so the failure is at least visible.

## Metadata: what embeddings can't do

Embeddings are good at meaning and bad at exact attributes like dates, IDs, names, and versions. Metadata covers those. It's useful in three ways:

1. **Filtering:** restrict the search to `department = 'Finance' AND year >= 2024`.
2. **Citations:** show "Source: Employee_Handbook.pdf, p. 14" with the answer.
3. **Ranking boosts:** prefer newer or official documents, or route numerical questions to tables.

Here's a case where filtering beats vector search outright. A user asks, "What's the parental leave policy in the 2025 handbook?" Your company has handbooks from 2021 through 2025, and their leave sections are nearly identical. The word "2025" barely moves the embedding, so pure vector search might return the 2022 version. A metadata filter fixes it exactly:

```python
results = vector_db.search(
    query_vector,
    top_k=3,
    filter={"doc_year": 2025, "doc_type": "handbook"},
)
```

At minimum, store these with every chunk: the source document, page or section, date, document type, and the embedding model name.

## Run it yourself

From the root of the [rag-scratch-to-speciality](https://github.com/sesha-2k3/rag-scratch-to-speciality/tree/part-02) repo:

```bash
uv sync
uv run python code/chunking/token_aware_scratch.py
```

The output reproduces the opening example, ending with `Question survived truncation? False`.

To see real tokenization and the near-1.0 similarity from silent truncation:

```bash
uv sync --group lib
uv run --group lib python code/chunking/token_aware_lib.py
```

The first run downloads the MiniLM model, which takes a moment.

Full code: [`token_aware_scratch.py`](https://github.com/sesha-2k3/rag-scratch-to-speciality/blob/part-02/code/chunking/token_aware_scratch.py) · [`token_aware_lib.py`](https://github.com/sesha-2k3/rag-scratch-to-speciality/blob/part-02/code/chunking/token_aware_lib.py)

## Takeaways

- Embedding models silently truncate input past their token limit, and the result still looks like a normal vector. Long chunks and pasted queries both lose information this way.
- Measure chunk sizes in tokens, using the embedding model's own tokenizer, with some headroom below the limit.
- Use one embedding model for both chunks and queries. Store the raw text and model name so you can re-embed later.
- Use metadata for what embeddings can't capture: exact dates, IDs, versions, and sources.

Want to test yourself? The repo's [chunking and embedding quiz](https://github.com/sesha-2k3/rag-scratch-to-speciality/blob/part-02/quizzes/01-chunking-embedding.md) covers Parts 1 and 2.

**Next:** Part 3: The query-time flow (coming soon)
