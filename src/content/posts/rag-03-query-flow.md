---
title: "RAG from Scratch, Part 3: The Query-Time Flow"
pubDatetime: 2026-10-21T00:30:00Z
featured: true
draft: false
kind: notes
category: AI
tags:
  - rag-series
  - rag
  - llm
description: "What actually happens between a user's question and the LLM's answer, why vector search always returns something, and how to design for 'no match'."
series: rag-from-scratch
seriesOrder: 3
---

**RAG from Scratch series:** [Part 0: Overview](/posts/rag-00-overview/) · [Part 1: Chunking](/posts/rag-01-chunking/) · [Part 2: Embeddings, tokens, and metadata](/posts/rag-02-embeddings/) · **Part 3: The query-time flow** · [Part 4: Evaluating retrieval](/posts/rag-04-evaluation/)

## Your vector database always has an answer

I built a tiny knowledge base with four chunks: three about a project called Atlas and one about a billing migration. Then I asked it something it knows nothing about:

```
Q: What is our parental leave policy?
   score=0.22  atlas_status.pdf: Project Atlas entered beta testing on Sept 12.
   score=0.00  atlas_roadmap.docx: The Atlas public launch is scheduled for Q1 2027.
   score=0.00  sync_notes.md: Atlas is blocked on the payments integration.
```

The search didn't say "nothing found." It returned the three nearest chunks, as it always does, and the top one has nothing to do with parental leave. Vector search ranks whatever is in the index. It has no idea whether any of it is relevant.

If those chunks went straight to the LLM, you'd be relying on the model to notice the context is useless and not improvise an answer from a beta-testing update. This post walks through the full query-time flow and shows where to stop that from happening.

## The LLM doesn't do the search

The most common misconception about RAG is that the LLM searches your documents. Usually, it doesn't. **Your application code searches first**, and the LLM only sees the results, pasted into its prompt as plain text:

```
User question
   → Embedding model (same model used at ingestion)
   → Vector database (nearest neighbors, top_k, optional metadata filter)
   → Prompt assembly (question + retrieved chunk TEXT)
   → LLM (answers from the given context)
   → Answer with citations
```

The LLM never sees embeddings or the database.

## The four steps

Let's follow the question "What's the current status of Project Atlas?" through the pipeline.

**1. Embed the question.** Use the same embedding model you used for the chunks, as Part 2 covered. With `all-MiniLM-L6-v2`, the question becomes 384 numbers.

**2. Search.** The vector database finds the stored vectors closest to the query vector, usually by cosine similarity. Real databases don't compare against every vector. They use an approximate-nearest-neighbor index, such as HNSW, to find close matches quickly. This is also where metadata filters like `project = "Atlas"` apply. Each hit comes back with a score, its text, and its metadata.

**3. Build the prompt from the chunks' text.** Not their vectors, which the LLM can't read:

```
Answer the question using only the context below.
If the answer isn't in the context, say you don't know.

Context:
[1] (Atlas_status_report.pdf, p.2) Project Atlas entered beta testing on Sept 12...
[2] (Atlas_roadmap.docx) The public launch is scheduled for Q1 2027...
[3] (Weekly_sync_notes.md) Atlas is blocked on the payments integration...

Question: What's the current status of Project Atlas?
```

The numbered sources let the LLM cite which chunk each claim came from.

**4. Generate.** The LLM answers using the text in its prompt.

In code, the whole flow is four lines:

```python
query_vec = embed_model.encode(question)
hits = vector_db.search(query_vec, top_k=3, filter={"project": "Atlas"})
context = "\n\n".join(h.text for h in hits)
answer = llm.generate(f"Context:\n{context}\n\nQuestion: {question}")
```

There's one common variation. In agentic or tool-calling setups, the LLM is given a `search_documents` tool and decides when to call it. Even then, your code runs steps 1 and 2 when the tool is called. Part 6 covers this.

## Building the pipeline from scratch

The scratch version needs an embedding model, a vector store, prompt assembly, and a threshold. To keep it dependency-free, the "embedding" is a hashed bag of words: each word adds 1 to one of 64 slots, and the vector is normalized. It matches words, not meaning, but it has the property that matters here, a fixed output size:

```python
def embed(text, dim=64):
    """Any input length -> exactly `dim` numbers (normalized)."""
    vec = [0.0] * dim
    for word in re.findall(r"\w+", text.lower()):
        if word not in STOPWORDS:
            vec[zlib.crc32(word.encode()) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]
```

Running it on a 2-word text and a 40-word text confirms this:

```
Vector size for a 2-word vs a 40-word text: 64 vs 64
```

The vector store keeps each chunk's vector, text, and metadata. Its search applies the filter, scores every row, and returns the top results:

```python
def search(self, query_vec, top_k=3, filter=None):
    """Always returns the top_k nearest rows, even if none are relevant."""
    if len(query_vec) != self.dim:
        raise ValueError(f"query dimension {len(query_vec)} does not match "
                         f"index dimension {self.dim}")
    rows = [r for r in self.rows
            if not filter or all(r[2].get(k) == v for k, v in filter.items())]
    scored = [(cosine(query_vec, vec), text, meta) for vec, text, meta in rows]
    return sorted(scored, key=lambda s: s[0], reverse=True)[:top_k]
```

Note the docstring. Nothing in `search` checks whether a result is any good. That's the next step's job:

```python
def answer(store, question, threshold=0.3, **search_kwargs):
    hits = store.search(embed(question, store.dim), **search_kwargs)
    hits = [h for h in hits if h[0] >= threshold]
    if not hits:
        return "I couldn't find that in the documents."
    return build_prompt(question, hits)
```

Here's the Atlas question with a `project = "Atlas"` filter:

```
Q: What is the status of Project Atlas launch?
   score=0.57  atlas_status.pdf: Project Atlas entered beta testing on Sept 12.
   score=0.41  atlas_roadmap.docx: The Atlas public launch is scheduled for Q1 2027.
   score=0.25  sync_notes.md: Atlas is blocked on the payments integration.
-> Prompt sent to the LLM:
Answer the question using only the context below.
If the answer isn't in the context, say you don't know.

Context:
[1] (atlas_status.pdf) Project Atlas entered beta testing on Sept 12.
[2] (atlas_roadmap.docx) The Atlas public launch is scheduled for Q1 2027.

Question: What is the status of Project Atlas launch?
```

Two chunks passed the 0.3 threshold and went into the prompt. The Hermes chunk never appeared at all, because the metadata filter removed it before scoring.

Look closely at the third hit, though. "Atlas is blocked on the payments integration" is arguably the most important status update of the three, and the threshold dropped it at 0.25. This is the trade-off with thresholds: one strict enough to reject the parental leave question can also cut a relevant chunk that happens to share fewer words with the query. A real embedding model would score it differently, since it captures meaning rather than word overlap, but the trade-off doesn't go away.

For the parental leave question, nothing passed:

```
-> No chunk passed the threshold: "I couldn't find that in the documents."
```

The top score of 0.22 is a hash collision in the toy embedding: two unrelated words happened to land in the same slot. Real embedding models have their own version of this problem. Unrelated text still gets a non-zero similarity score, which is exactly why the threshold has to be calibrated rather than assumed.

## What if the query doesn't "match" the chunks?

A common worry is that a short question can't be compared with a long chunk. Here's what actually happens in each case:

| Situation | What happens |
|---|---|
| Question much shorter or longer than the chunks | **Fine.** The output dimension is fixed. Only queries past the token limit are a problem, through silent truncation (Part 2). |
| Query vector has a different dimension | **Rejected** with a dimension-mismatch error. Similarity between vectors of different lengths is undefined. |
| Different model, same dimension | **No error, meaningless results.** |
| No relevant content exists | **Fails silently.** Vector search always returns the top_k nearest, even if barely related. |

The scratch code demonstrates the second row by searching with a 1,536-number vector, the size `text-embedding-3-small` produces:

```
Dimension mismatch is rejected: query dimension 1536 does not match index dimension 64
```

The last row is the dangerous one, because nothing tells you it happened.

## Designing for "no match"

Use two defenses together.

**A similarity threshold** drops weak hits. If nothing survives, skip the LLM entirely and say nothing was found:

```python
hits = [h for h in hits if h.score >= 0.5]
if not hits:
    return "I couldn't find anything about that in the project documents."
```

The right cutoff depends on the embedding model, because each model's scores fall in a different range. Calibrate it on real queries: collect questions your documents can answer and questions they can't, and look at where their scores fall. Part 4 covers how to measure this properly.

**A prompt instruction** catches what the threshold misses: "If the answer isn't in the context, say you don't know." This matters for weak-but-passing hits, which the threshold lets through but the LLM can still recognize as unhelpful.

## The same thing with a real embedding model

The library version swaps the toy embedding for `all-MiniLM-L6-v2` and the vector store for a NumPy matrix:

```python
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")  # same model for chunks and queries

# Ingestion, done once. Normalized vectors make dot product equal cosine.
index = model.encode([c["text"] for c in chunks], normalize_embeddings=True)

def retrieve(question, top_k=3, threshold=0.3, project=None):
    q = model.encode(question, normalize_embeddings=True)    # 1. embed
    scores = index @ q                                        # 2. similarity
    order = np.argsort(-scores)
    hits = [(float(scores[i]), chunks[i]) for i in order
            if project is None or chunks[i]["project"] == project][:top_k]
    return [h for h in hits if h[0] >= threshold]             # 3. threshold
```

`index @ q` scores every chunk in one matrix operation. That's fine for a few thousand chunks. Beyond that, you'd move to a vector database with an approximate-nearest-neighbor index.

The script ends by printing the assembled prompt instead of calling an LLM, so it runs without an API key. Swap the `print` for your LLM client's call to complete the pipeline.

## Run it yourself

From the root of the [rag-scratch-to-speciality](https://github.com/sesha-2k3/rag-scratch-to-speciality/tree/part-03) repo:

```bash
uv sync
uv run python code/retrieval/query_pipeline_scratch.py
```

The output includes both questions above and the dimension-mismatch error.

For the real embedding model:

```bash
uv sync --group lib
uv run --group lib python code/retrieval/query_pipeline_lib.py
```

Compare its scores with the scratch version's. In particular, check where the "blocked on the payments integration" chunk lands with an embedding that understands meaning.

Full code: [`query_pipeline_scratch.py`](https://github.com/sesha-2k3/rag-scratch-to-speciality/blob/part-03/code/retrieval/query_pipeline_scratch.py) · [`query_pipeline_lib.py`](https://github.com/sesha-2k3/rag-scratch-to-speciality/blob/part-03/code/retrieval/query_pipeline_lib.py)

## Takeaways

- Your application code does the search. The LLM only sees retrieved chunk text pasted into its prompt.
- Query and chunk lengths don't need to match, because embedding dimensions are fixed. Mismatched models or dimensions are the real problem.
- Vector search always returns its top_k results, relevant or not. Without a check, irrelevant chunks reach the LLM silently.
- Combine a calibrated similarity threshold with a "say you don't know" instruction, and remember that thresholds can cut relevant chunks too.

Want to test yourself? The repo's [chunking and embedding quiz](https://github.com/sesha-2k3/rag-scratch-to-speciality/blob/part-03/quizzes/01-chunking-embedding.md) covers Parts 1 through 3.

**Next:** [Part 4: Evaluating retrieval](/posts/rag-04-evaluation/)
