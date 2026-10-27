---
title: "RAG from Scratch, Part 6: Advanced Retrieval"
pubDatetime: 2026-10-26T02:00:00Z
featured: true
draft: false
tags:
  - rag-series
  - rag
  - llm
  - search
description: "Hybrid search with Reciprocal Rank Fusion, rerankers, query transformation, small-to-big and contextual retrieval, GraphRAG, and agentic RAG, plus how to choose between them."
series: rag-from-scratch
seriesOrder: 6
---

**RAG from Scratch series:** [Part 0: Overview](/posts/rag-00-overview/) · [Part 1: Chunking](/posts/rag-01-chunking/) · [Part 2: Embeddings, tokens, and metadata](/posts/rag-02-embeddings/) · [Part 3: The query-time flow](/posts/rag-03-query-flow/) · [Part 4: Evaluating retrieval](/posts/rag-04-evaluation/) · [Part 5: BM25](/posts/rag-05-bm25/) · **Part 6: Advanced retrieval**

## Adding scores doesn't combine them

Part 5 ended with a problem. Vector search finds "paid time off" when you ask about vacation. BM25 finds "E-4012" when you ask about E-4012. You want both, so the obvious move is to run both searches and add their scores:

```python
cosine   = {"A": 0.82, "B": 0.80, "C": 0.55}
bm25_raw = {"C": 14.2, "A": 6.1,  "B": 0.0}
naive = {d: cosine[d] + bm25_raw[d] for d in cosine}
```

```
Naive score addition: ['C', 'A', 'B'] <- order is just the BM25 order
```

Cosine similarities sit roughly between 0.2 and 0.9. BM25 scores are unbounded, as Part 5 showed, so a BM25 score of 14.2 swamps any cosine value. The "combined" ranking is just BM25's ranking, and the vector search might as well not have run. Normalizing scores to a common range sounds like a fix, but it's fragile, because each query produces a different distribution of scores.

This final part starts with the standard solution to that problem, then works through the other techniques this series has been pointing toward: rerankers, query transformation, small-to-big and contextual retrieval, GraphRAG, and agentic RAG. It ends with how to choose between them.

## Hybrid search with Reciprocal Rank Fusion

Reciprocal Rank Fusion (RRF) sidesteps the scale problem by ignoring scores entirely. It only uses each document's **rank** in each list:

```
score(d) = Σ over result lists of  1 / (k + rank of d in that list)
```

k is a constant, usually 60, from the paper that introduced RRF. A large k keeps the gap between rank 1 and rank 2 small, so no single list dominates.

The implementation is a few lines:

```python
from collections import defaultdict

def rrf(ranked_lists, k=60):
    """Reciprocal Rank Fusion: uses only ranks, ignores raw scores."""
    scores = defaultdict(float)
    for ranking in ranked_lists:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
```

Suppose vector search returns `[A, B, C, D]` and BM25 returns `[C, A, E, F]`:

| Document | Vector rank | BM25 rank | RRF score |
|---|---|---|---|
| A | 1 | 2 | 1/61 + 1/62 = **0.03252** |
| C | 3 | 1 | 1/63 + 1/61 = **0.03227** |
| B | 2 | — | 1/62 = 0.01613 |
| E | — | 3 | 1/63 = 0.01587 |
| D, F | 4 | 4 | 1/64 = 0.01562 each |

The script's output matches:

```
RRF (k=60):
  A: 0.03252
  C: 0.03227
  B: 0.01613
  E: 0.01587
  D: 0.01562
  F: 0.01562
```

A and C rise to the top because **both** methods found them. B was vector search's second choice, but BM25 never saw it, so it drops below them. That's the behavior you want: agreement between two very different retrieval methods is strong evidence of relevance.

You rarely need to write this yourself. Weaviate, Qdrant, Elasticsearch, and OpenSearch all support hybrid search natively. A common pattern is to send the fused top 20–30 results to a reranker.

## Rerankers

Vector search uses a **bi-encoder**: the query and each chunk are embedded *separately*, and compared by cosine similarity. That's what makes it fast, because every chunk is embedded once in advance. It's also why it's imprecise, because the model never sees the query and the chunk together.

A **cross-encoder** reads each query–chunk pair *together* and outputs a relevance score. It's much more accurate, and much too slow to run over millions of chunks.

So you combine them. Retrieve broadly with the fast method, then rerank the short list with the accurate one:

```python
from sentence_transformers import CrossEncoder

candidates = vector_db.search(query_vector, top_k=30)          # fast recall

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = reranker.predict([(user_query, c.text) for c in candidates])
top_chunks = [c for _, c in sorted(zip(scores, candidates), reverse=True)][:5]
```

This is the fix for the diagnostic from Part 4: if recall@10 is high but recall@3 is low, retrieval is finding the right chunks but ranking them poorly. A reranker reorders the same candidates, which is why nDCG, not recall, is the metric that shows whether it helped.

A related technique, **Maximal Marginal Relevance (MMR)**, picks chunks that are relevant *and* different from each other, so the LLM doesn't get five near-duplicates.

## Query transformation

Sometimes the problem is the question, not the index.

**Query rewriting** turns a follow-up like "what about for contractors?" into a standalone question using the chat history. It's also the fix for pasted content from Part 2.

**Multi-query** asks an LLM for several paraphrases of the question, searches with each, and fuses the results with RRF. It catches documents that one phrasing misses.

**HyDE (Hypothetical Document Embeddings)** asks the LLM to write a *hypothetical answer*, then embeds that instead of the question. It works because answers resemble answer chunks more than questions do. It also fails in predictable ways. When the LLM doesn't know your domain, it might invent an answer about NASA's Atlas rocket and steer retrieval away from your internal Project Atlas docs. For exact-term queries, the hypothetical answer may not include "E-4012" at all. And it adds an LLM call before every search.

Treat all of these as optional, and test each one on your evaluation set before keeping it.

## Query decomposition

Some questions can't be answered with one search.

| | Parallel | Sequential (multi-hop) |
|---|---|---|
| Example | "Compare the budgets of Atlas and Hermes" | "What was the previous project of the person who leads Atlas?" |
| Why one search fails | One vector for two topics; the top 3 may all be about Atlas | The chunk with the answer mentions "Priya Rao," not Atlas |
| Method | Search "Atlas budget" and "Hermes budget" separately, then combine | Step 1 finds who leads Atlas; step 2 searches her previous projects |
| Latency | One round | Several rounds |
| Risk | Splitting simple questions unnecessarily | A wrong intermediate answer derails every later step |

## Self-query retrieval

"Show me Atlas status reports from last month" mixes a topic with filters. Embeddings have no concept of "last month." Self-query retrieval has an LLM translate the question into a search query plus metadata filters, given the current date:

```json
{"query": "project status report",
 "filter": {"project": "Atlas", "doc_type": "status_report",
            "date": {"gte": "2026-09-01", "lte": "2026-09-30"}}}
```

This is where Part 2's metadata pays off. Give the LLM your metadata schema and the current date, validate its output before running the search, and fall back to looser filters if nothing matches. LangChain's `SelfQueryRetriever` and LlamaIndex's auto-retrieval implement this.

## Small-to-big retrieval

Part 1 described a tension in chunk size: small chunks match precisely but lose context, while large chunks keep context but blur the embedding. Part 4 showed that bigger chunks can even raise recall while making answers worse.

Small-to-big retrieval, also called parent document retrieval, resolves this by using two sizes. Embed small **child** chunks, such as sentences or about 200 tokens, each storing a `parent_id`. Search over the children, then send the LLM their **parent** section, about 1,500 tokens. You get precise matching and enough context.

The costs: more tokens per query, duplicate parents to remove when several children share one, and the question of how big a parent should be. Sections based on document headings work well. Two variants of the idea are **sentence window** retrieval, which returns a few sentences around each match, and **auto-merging**, which swaps in the parent when enough of its children match.

## Contextual retrieval

Consider this chunk: "Revenue grew 3% over the previous quarter." Whose revenue? Which quarter? Chunking stripped the context away.

Anthropic's [contextual retrieval](https://www.anthropic.com/news/contextual-retrieval) technique, published in 2024, fixes this at indexing time. For each chunk, give an LLM the whole document plus the chunk, and ask for a short situating context, such as "This chunk is from ACME Corp's Q2 2026 earnings report, financial performance section." Prepend that context to the chunk, then use it for both the embedding **and** the BM25 index.

In Anthropic's experiments, contextual embeddings alone reduced the top-20 retrieval failure rate by 35%. Adding contextual BM25 brought the reduction to 49%, and adding a reranker on top brought it to 67%.

The cost is one LLM call per chunk, each including the full document. Prompt caching makes this practical, because the document is cached once and reused for each of its chunks. You also need to rerun it whenever documents change.

There's a free version that captures much of the benefit: prepend each chunk's document title and heading path, such as "Employee Handbook > Leave > Parental Leave." That costs nothing and is the first thing to try.

## GraphRAG

Vector RAG has a blind spot for questions about the corpus as a whole. "What are the main recurring risks across all our projects?" can't be answered well from the 5 most similar chunks out of 10,000.

GraphRAG addresses this with a heavier indexing step. An LLM extracts entities and relationships from every chunk, those are built into a graph, the graph is clustered into communities of related entities, and an LLM summarizes each community at several levels of detail. Global questions are then answered by combining the community summaries, which together cover the whole corpus.

The graph can also trace connections that no single document states: an outage hit the payments service, Atlas depends on the payments service, so Atlas was delayed.

The costs are real. Indexing is expensive, often far more than embedding. Incremental updates are hard. Extraction errors spread silently through the graph. And for simple lookups, it's overkill.

For questions about structured data, such as "How many projects went over budget in 2025?", neither vector RAG nor GraphRAG is the right tool. Use text-to-SQL against the database instead.

## Agentic RAG

In agentic RAG, the LLM is given tools, such as `search_docs` and `query_database`, and runs a loop: search, read the results, decide what's missing, search again, and finally answer. This is the tool-calling variation from Part 3. As there, your code still runs each search when the LLM calls the tool.

It's worth it for multi-hop questions where the path isn't known in advance, exploratory research, combining several different kinds of sources, and checking its own answers.

It also brings new failure modes. Latency and cost grow, from about 2 seconds per answer to 20–40. The path through the tools is unpredictable. The agent can loop or stop too early. And errors compound, since each step builds on the last.

To keep it under control, cap the number of iterations and tokens, log every trace, and evaluate the path the agent took as well as its final answer. A lighter alternative is a **router**: classify each question, and send each type to a fixed pipeline built for it.

## The same thing with LangChain

LangChain's `EnsembleRetriever` implements hybrid search with weighted RRF:

```python
from langchain_community.retrievers import BM25Retriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import EnsembleRetriever  # langchain.retrievers before 1.0

bm25 = BM25Retriever.from_texts(texts)
bm25.k = 3

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector = InMemoryVectorStore.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 3})

hybrid = EnsembleRetriever(retrievers=[bm25, vector], weights=[0.5, 0.5])
```

The script runs two queries through BM25, vector search, and the hybrid retriever, and prints each one's top hit. "Error code E-4012" is the case BM25 is built for, and "how much vacation do I get?" is the case vector search is built for. Run it to see which retriever wins each query, and check that the hybrid retriever gets both right.

## Choosing techniques

Every technique in this post solves a specific problem. Start from the symptom you're seeing:

| Symptom | Cause | Technique |
|---|---|---|
| Part numbers and codes are missed | Embeddings blur exact strings | Hybrid search |
| "Compare X and Y" only covers X | One vector for two topics | Parallel decomposition |
| Vague answers to questions about the whole corpus | Top-k sees only a few chunks | GraphRAG, or summarizing across documents |
| Chunks say "this project" without naming it | Chunking removed context | Heading prefixes, then contextual retrieval |
| The right chunks land at ranks 4–10 | A ranking problem, not a finding problem | Reranker |
| Pasted text breaks retrieval | Truncation and dilution | Query rewriting |

If you're starting from a basic pipeline, a sensible order is:

1. **Title and heading prefixes on chunks.** Almost free.
2. **Hybrid search plus a reranker.** Low risk, with broad gains.
3. **Query decomposition behind a router,** if your users ask compound questions.
4. **Solutions for corpus-wide questions,** such as GraphRAG, only if those questions are common.

Most importantly, **add one technique at a time and measure each** against your evaluation set from Part 4. Every technique here adds cost or latency, and some will make your particular system worse.

## Run it yourself

From the root of the [rag-scratch-to-speciality](https://github.com/sesha-2k3/rag-scratch-to-speciality/tree/part-06) repo:

```bash
uv sync
uv run python code/retrieval/rrf_scratch.py
```

For the LangChain version:

```bash
uv sync --group lib
uv run --group lib python code/retrieval/rrf_lib.py
```

Full code: [`rrf_scratch.py`](https://github.com/sesha-2k3/rag-scratch-to-speciality/blob/part-06/code/retrieval/rrf_scratch.py) · [`rrf_lib.py`](https://github.com/sesha-2k3/rag-scratch-to-speciality/blob/part-06/code/retrieval/rrf_lib.py)

## Takeaways

- Don't add vector and BM25 scores, because BM25's scale swamps cosine. Reciprocal Rank Fusion combines rankings instead, and rewards documents both methods found.
- Retrieve broadly with fast bi-encoders, then rerank a short list with an accurate cross-encoder.
- Query rewriting, multi-query, HyDE, decomposition, and self-query fix problems in the question rather than the index.
- Small-to-big and contextual retrieval fix problems created by chunking. Start with free heading prefixes.
- GraphRAG and agentic RAG handle questions plain retrieval can't, at a real cost in money, latency, and predictability.
- Choose techniques from symptoms, add them one at a time, and measure each.

Want to test yourself? The repo's [advanced retrieval quiz](https://github.com/sesha-2k3/rag-scratch-to-speciality/blob/part-06/quizzes/03-advanced-retrieval.md) covers Parts 5 and 6.

## Wrapping up the series

That's the end of RAG from Scratch. Across six parts, we built every stage of a retrieval pipeline by hand:

- **Part 1:** splitting documents into chunks, and why a "50-character" chunk was really 47.
- **Part 2:** embedding them, and how silent truncation can make a user's question vanish.
- **Part 3:** the query-time flow, and why vector search always returns *something*.
- **Part 4:** measuring retrieval properly, and why 82% vs 78% wasn't a real difference.
- **Part 5:** BM25, for the exact matches embeddings blur.
- **Part 6:** combining and improving all of it.

If one idea carries through every part, it's this: **don't trust what you haven't measured.** Print your chunks, count your tokens, calibrate your thresholds, check your nDCG's ideal ranking, and test whether a gain is real before you ship it.

All the code is in [rag-scratch-to-speciality](https://github.com/sesha-2k3/rag-scratch-to-speciality). If you spot a mistake or have a question, open an issue there. Thanks for reading.
