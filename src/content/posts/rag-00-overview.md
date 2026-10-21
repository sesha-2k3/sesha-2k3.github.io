---
title: "RAG from Scratch, Part 0: What This Series Covers"
pubDatetime: 2026-10-16T00:00:00Z
featured: false
draft: false
tags:
  - rag-series
  - rag
  - llm
description: "A six-part series on Retrieval-Augmented Generation, built from first principles with runnable code: chunking, embeddings, the query flow, evaluation, BM25, and advanced retrieval."
---

**RAG from Scratch series:** **Part 0: Overview** · [Part 1: Chunking](/posts/rag-01-chunking/)

Retrieval-Augmented Generation (RAG) sounds simple: find the relevant text, give it to an LLM, get a grounded answer. In practice, every step hides a decision that can quietly break the system. How big should a chunk be? What happens when a query is longer than the embedding model allows? Is a 4-point accuracy gain real, or noise?

This series is my attempt to answer those questions properly, by building each piece myself before reaching for a library.

## What's in the series

| Part | Topic | What you'll learn |
|---|---|---|
| 1 | [Chunking](/posts/rag-01-chunking/) | Why we split documents, five chunking strategies, overlap, and choosing a chunk size |
| 2 | [Embeddings, tokens, and metadata](/posts/rag-02-embeddings/) | Why characters aren't tokens, silent truncation, and what metadata to store |
| 3 | The query-time flow | What happens between a user's question and the LLM's answer, and designing for "no match" |
| 4 | Evaluating retrieval | Precision, recall, MRR, nDCG, test sets, LLM-as-judge, and whether 82% vs 78% is significant |
| 5 | BM25 | Why keyword search still matters, the formula, and a worked example by hand |
| 6 | Advanced retrieval | Hybrid search, Reciprocal Rank Fusion, rerankers, query transformation, and more |

Parts without links are coming soon.

## How the code works

All the code lives in [rag-scratch-to-speciality](https://github.com/sesha-2k3/rag-scratch-to-speciality) on GitHub. Every example comes in two versions:

- **`*_scratch.py`** uses only the Python standard library. No packages, runs offline. Read these to understand the mechanics.
- **`*_lib.py`** does the same thing with real libraries (LangChain, sentence-transformers, rank_bm25, ranx, statsmodels). Use these as templates for real projects.

Each worked example in the series has an expected output, so you can check my claims by running the code.

## Setup

The project uses [uv](https://docs.astral.sh/uv/). Clone the repo and run everything from its root:

```bash
git clone https://github.com/sesha-2k3/rag-scratch-to-speciality.git
cd rag-scratch-to-speciality

# Scratch versions: standard library only
uv sync
uv run python code/chunking/fixed_recursive_scratch.py

# Library versions: install the lib group first
uv sync --group lib
uv run --group lib python code/chunking/fixed_recursive_lib.py
```

## Where this started

The series began from Jesvin K Justin's Medium article, [From Zero to RAG: The Art of Document Chunking and Embedding for RAG](https://medium.com/@jesvinkjustin/from-zero-to-rag-the-art-of-document-chunking-and-embedding-for-rag-d9764695cc46). I worked through it, checked its examples by running them, and extended it with evaluation, keyword search, and advanced retrieval.

**Next:** [Part 1: Chunking](/posts/rag-01-chunking/)
