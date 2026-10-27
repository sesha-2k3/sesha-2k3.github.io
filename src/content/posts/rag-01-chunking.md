---
title: "RAG from Scratch, Part 1: Chunking"
pubDatetime: 2026-10-16T00:05:00Z
featured: true
draft: false
tags:
  - rag-series
  - rag
  - llm
description: "Why a '50-character' chunk turned out to be 47, and how fixed-size and recursive chunking actually work, implemented from scratch in Python."
series: rag-from-scratch
seriesOrder: 1
---

**RAG from Scratch series:** [Part 0: Overview](/posts/rag-00-overview/) · **Part 1: Chunking** · [Part 2: Embeddings, tokens, and metadata](/posts/rag-02-embeddings/)

## The 47-character chunk

The tutorial I learned chunking from showed a fixed-size splitter cutting text every 50 characters. Its first chunk was:

```
"Artificial Intelligence is transforming industr"
```

That looks plausible until you count it. It's 47 characters. A real 50-character cut of the same text gives:

```
"Artificial Intelligence is transforming industries"
```

It's a small slip, and tutorial outputs are usually illustrations rather than real runs. But it taught me the most useful habit in this whole topic: **never trust a chunker you haven't printed.**

```python
for i, c in enumerate(chunks):
    print(i, len(c), repr(c))
```

The rest of this post builds two chunkers from scratch so you can see exactly what they do.

## Why chunk at all?

An embedding model turns text into a fixed-length vector, for example 384 or 1,536 numbers, no matter how much text goes in. If you embed a 100-page document as one vector, that vector is an average of every topic in the document. It ends up loosely similar to many queries and strongly similar to none.

There are three more reasons to split:

- **Model limits.** Embedding models silently truncate input past their token limit. Most of a long document would simply never be embedded.
- **Cost and context.** Sending whole documents to the LLM for every question is slow, expensive, and can overflow the context window.
- **Precision.** Most questions are answered by a few sentences. Burying them in 100 pages makes the LLM more likely to miss them, especially in the middle of a long context.

Chunking turns one blurry vector into many sharp ones, each about a single topic.

## Five ways to chunk

| Strategy | How it splits | Strength | Weakness |
|---|---|---|---|
| Fixed-size | Every N characters or tokens | Simple, predictable | Cuts mid-sentence, ignores meaning |
| Recursive | Tries paragraphs, then lines, then words, then characters | Respects natural boundaries | Still driven by size, not meaning |
| Document-based | By structure: headings, sections, code blocks, tables | Follows how the author organized the content | Needs a layout-aware parser |
| Semantic | Embeds sentences and cuts where neighbors stop being similar | Splits at topic shifts | Extra embedding cost |
| Agentic | An LLM extracts standalone statements, then groups them | Groups related ideas from anywhere in the text | Many LLM calls; slow, costly, non-deterministic |

This post implements the first two, since they're the foundation the others build on.

## Fixed-size chunking from scratch

The simplest chunker cuts every `chunk_size` characters. To avoid losing facts that land on a boundary, it steps forward by `chunk_size - overlap`, so the end of each chunk repeats at the start of the next:

```python
def fixed_size_chunks(text, chunk_size, overlap):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    step = chunk_size - overlap
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        if start + chunk_size >= len(text):
            break
        start += step
    return chunks
```

The `overlap >= chunk_size` check matters: without it, `step` would be zero or negative, and the loop would never advance.

Here's the output on a short four-sentence text with `chunk_size=50` and `overlap=10`:

```
chunk 1 (50 chars): 'Artificial Intelligence is transforming industries'
chunk 2 (50 chars): 'industries. It has applications in healthcare, fin'
chunk 3 (50 chars): 'hcare, finance, and education.\n\nMachine learning m'
chunk 4 (50 chars): 'learning models learn patterns from data.\nDeep lea'
chunk 5 (50 chars): '.\nDeep learning uses neural networks with many lay'
chunk 6 (14 chars): 'h many layers.'
```

You can see both the overlap working ("industries" appears at the end of chunk 1 and the start of chunk 2) and the core problem: chunks end in the middle of words like "fin" and "lay". The embedding of chunk 3 has to make sense of "hcare, finance".

## Recursive chunking from scratch

Recursive chunking fixes this by trying the most natural separator first and only falling back to cruder ones when a piece is still too big. The order is: paragraph breaks (`\n\n`), then line breaks (`\n`), then spaces, and only as a last resort, individual characters.

```python
def _split(text, chunk_size, separators):
    if len(text) <= chunk_size:
        return [text]
    sep, rest = separators[0], separators[1:]
    if sep == "":  # last resort: hard cut by characters
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    parts = text.split(sep)
    out = []
    for i, part in enumerate(parts):
        piece = part + (sep if i < len(parts) - 1 else "")  # keep the separator
        if len(piece) <= chunk_size:
            out.append(piece)
        else:
            out.extend(_split(piece, chunk_size, rest))  # recurse with next separator
    return out
```

Splitting alone leaves many tiny pieces, so a second step merges neighbors back together until they approach `chunk_size`:

```python
def _merge(pieces, chunk_size):
    chunks, current = [], ""
    for piece in pieces:
        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            if current.strip():
                chunks.append(current.strip())
            current = piece
    if current.strip():
        chunks.append(current.strip())
    return chunks
```

On the same text with `chunk_size=60`:

```
chunk 1 (58 chars): 'Artificial Intelligence is transforming industries. It has'
chunk 2 (51 chars): 'applications in healthcare, finance, and education.'
chunk 3 (49 chars): 'Machine learning models learn patterns from data.'
chunk 4 (52 chars): 'Deep learning uses neural networks with many layers.'
```

Every chunk now ends on a whole word, and chunks 3 and 4 are clean sentences, because the line break was a natural place to cut. Chunk 1 still splits a sentence, because the first paragraph is longer than 60 characters and had to fall back to splitting on spaces. That's the trade-off: recursive chunking respects boundaries, but it's still driven by size, not meaning.

My version leaves out overlap to keep the logic readable. The library version below includes it.

## The same thing with LangChain

In practice, you'd use LangChain's splitters, which implement the same ideas:

```python
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

fixed = CharacterTextSplitter(separator="", chunk_size=50, chunk_overlap=10)
recursive = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""], chunk_size=60, chunk_overlap=10
)

chunks = recursive.split_text(text)
```

Expect small differences from the scratch output. LangChain strips whitespace from chunk edges and handles overlap during merging, so chunk lengths vary a little. Print the chunks and compare them yourself.

## Overlap: cheap insurance

Overlap exists for one failure in particular. Without it, a policy document might split like this:

```
"...employees are entitled to 12 days of" | "paid sick leave per calendar year..."
```

Neither chunk answers "How many sick days do I get?" With overlap, the second chunk begins "to 12 days of paid sick leave...", and the full fact survives in one place.

The cost is redundancy: more storage, more embedding calls, and the same sentence can be retrieved twice. A reasonable starting point is 10–20% of the chunk size.

## Choosing a chunk size

Chunk size fails in two opposite directions:

- **Too small, and chunks lose context.** "It was increased to 20% in 2023." Increased what? The embedding can't capture meaning that isn't in the text.
- **Too large, and topics bleed together.** A chunk covering tax rates, hiring, and office locations gets an averaged vector that matches every query weakly, and sends the LLM a lot of irrelevant text.

A common starting point is 200–500 tokens with some overlap. But the right answer depends on your documents and your users' questions, which is why Part 4 of this series covers how to measure it rather than guess.

## Tables and images

Tables and images don't embed well as raw content, because users ask questions in natural language. The usual fix is to embed an LLM-written summary of the table or image, store the raw content alongside it, and pass the raw content to the LLM when the summary is retrieved. You find with the summary and answer with the data. This is called the multi-vector retriever pattern.

## Run it yourself

From the root of the [rag-scratch-to-speciality](https://github.com/sesha-2k3/rag-scratch-to-speciality/tree/part-01) repo:

```bash
uv sync
uv run python code/chunking/fixed_recursive_scratch.py
```

The last lines of the output reproduce the opening example:

```
The blog's 'Chunk 1' is 47 chars, not 50.
A real 50-char cut gives: 'Artificial Intelligence is transforming industries'
```

To run the LangChain version:

```bash
uv sync --group lib
uv run --group lib python code/chunking/fixed_recursive_lib.py
```

Full code: [`fixed_recursive_scratch.py`](https://github.com/sesha-2k3/rag-scratch-to-speciality/blob/part-01/code/chunking/fixed_recursive_scratch.py) · [`fixed_recursive_lib.py`](https://github.com/sesha-2k3/rag-scratch-to-speciality/blob/part-01/code/chunking/fixed_recursive_lib.py)

## Takeaways

- Chunking exists because one vector can't represent a whole document well, and models truncate long input anyway.
- Fixed-size chunking is simple but cuts through words and sentences. Recursive chunking cuts at natural boundaries but is still size-driven.
- Overlap protects facts that land on a boundary, at the cost of some redundancy.
- Always print your chunks, including the length of each one.

**Next:** [Part 2: Embeddings, tokens, and metadata](/posts/rag-02-embeddings/)
