---
title: "HNSW Indexing in Vector Databases: How It Works and Why It Matters for RAG"
pubDatetime: 2026-10-25T00:00:00Z
featured: false
draft: false
tags:
  - rag
  - llm
  - vector-databases
description: "How the HNSW index makes vector search fast: the layered graph, how it's built, how a query walks it, and what goes wrong in production, with a worked example you can check by hand."
---

In a RAG pipeline, documents are split into chunks, each chunk is turned into an embedding vector, and the vectors are stored in a vector database. At query time, retrieval means finding the stored vectors closest to the question's vector. With a few thousand chunks, you can compare the question against every one of them. With a few million, that gets slow, and this is where most vector databases quietly hand the job to an index called HNSW.

Most RAG tutorials treat the index as a box you never open. That's fine until retrieval starts missing chunks you know are in the database, or results get worse after a month of document updates. Both problems come from how HNSW works, so this post opens the box.

**How to read this post:** if you're new to vector search, read it top to bottom; the worked example is the heart of it. If you've used vector databases before, skip to [How layers are assigned](#how-layers-are-assigned), [The stopping rule](#the-stopping-rule-and-ef_search), and [HNSW in production](#hnsw-in-production).

## Where the index sits in RAG

A RAG pipeline has two halves:

| Stage | When it runs | What happens |
|---|---|---|
| Chunking | Ingestion | Split documents into chunks |
| Embedding | Ingestion | Turn each chunk into a vector |
| **Indexing** | **Ingestion** | **The vector database builds the HNSW graph from the vectors** |
| Query embedding | Every question | Turn the question into a vector with the same model |
| **Search** | **Every question** | **Walk the graph to find the closest chunks** |
| Generation | Every question | Put those chunks in the prompt and call the LLM |

Two things are worth stating plainly, because they're easy to blur:

1. **HNSW is built from the embeddings, not from the text.** The embedding model knows nothing about graphs. The database takes the vectors it produced and builds the graph as a separate step. Each chunk's vector becomes one node in a single graph covering your whole collection, and the chunk's text and metadata are stored alongside the node.
2. **HNSW doesn't replace semantic similarity; it's a faster way of computing it.** "Relevant" still means "closest embedding." HNSW only decides which chunks are worth comparing. Swap it for brute force and you'd get nearly the same chunks back, just more slowly. So if your retrieval is poor because of bad chunking or a weak embedding model, HNSW can't fix it.

## Why not just compare everything?

The exact approach, called brute force or flat search, computes the distance from the query to every stored vector, sorts them, and takes the top k. For 1 million chunks with 768-dimensional embeddings, that's about 768 million multiply-adds per question. That's manageable for one query on a fast machine, but heavy under real traffic, and it grows linearly with your corpus.

HNSW trades a little accuracy for a lot of speed. It's an **approximate nearest neighbour (ANN)** index: it usually finds the true closest chunks, but not always. The fraction it finds is called **recall**. If brute force says the top 10 chunks are a certain set and HNSW returns 9 of them, recall is 0.9.

In RAG, recall has a direct cost. If the chunk that answers the question isn't retrieved, the LLM never sees it, and it either answers from weaker context or makes something up.

## The intuition: airports, highways, streets

Picture every chunk as a point on a map, where chunks with similar meaning sit close together. Your question is a new point on the same map, and you want to find the points nearest to it.

If you were travelling to a specific house in another country, you wouldn't walk street by street. You'd fly to the right city, drive the highways to the right neighbourhood, then walk the local streets to the door.

HNSW (Hierarchical Navigable Small World) organizes your chunks the same way, as a stack of graphs:

```
Layer 2   •───────────────────────•                  few nodes, long links  (airports)
          │                       │
Layer 1   •──────•────────•───────•────────•         more nodes             (highways)
          │      │        │       │        │
Layer 0   •─•─•─•─•─•─•─•─•─•─•─•─•─•─•─•─•─•        every chunk            (streets)
```

Every chunk is on layer 0. A random few are also on layer 1, fewer still on layer 2, and so on. A node on a high layer also appears on every layer below it. Each node links to nearby nodes on each of its layers.

A search starts at a fixed entry point on the top layer, takes big hops toward the query, drops down a layer, takes smaller hops, and repeats until it reaches layer 0, where it does a careful local search. The top layers let it skip most of the map, so only a small part of the work happens in the crowded bottom layer.

## A worked example you can check by hand

To make every step traceable, the example uses a tiny help-center knowledge base with 2D vectors and ordinary straight-line distance. Real systems use hundreds or thousands of dimensions and usually cosine distance, but the steps are identical.

Settings: `M = 2` (each new node links to 2 neighbours), at most 3 links per node on layer 0, at most 2 on layer 1, and `ef_construction = 4`.

| Chunk | Text | Vector |
|---|---|---|
| A | Reset your password | (1, 1) |
| B | Change account email | (2, 1) |
| C | Set up two-factor login | (2, 2) |
| D | Refund policy | (7, 2) |
| E | Cancel subscription | (8, 3) |
| F | Update billing card | (7, 4) |

Account topics cluster on the left of the map and billing topics on the right.

### Building the graph

Nodes are inserted one at a time, in order. Each gets a randomly chosen top layer (more on how [below](#how-layers-are-assigned)); here, A and D land on layer 1 and the rest stay on layer 0.

Inserting a node uses the same search a query does: find where the new vector belongs, then link it to its nearest neighbours.

| Insert | Top layer | What happens |
|---|---|---|
| A | 1 | First node, so it becomes the entry point. Nothing to link to. |
| B | 0 | Only A exists, so B links to A. |
| C | 0 | B is at 1.0 and A at 1.41. C links to both. |
| D | 1 | On layer 1, D links to A, the only node there. On layer 0, the search finds C (5.0) and B (5.10), so D links to them. |
| E | 0 | Layer 1: from A (7.28 away), hop to D (1.41). Layer 0 from D: closest are D (1.41) and C (6.08). E links to both. C now has 4 links, over the cap of 3, so C drops its farthest, E. |
| F | 0 | Layer 1: A to D. Layer 0: closest are E (1.41) and D (2.0). F links to both. D now has 4 links, so it drops its farthest, B. |

The finished graph:

| Node | Layer 0 links | Layer 1 links |
|---|---|---|
| A | B, C | D |
| B | A, C, D | |
| C | B, A, D | |
| D | C, E, F | A |
| E | D, C, F | |
| F | E, D | |

Two details matter later. First, pruning left some links one-way: B points to D, but D no longer points back. Second, the only layer-1 link is A to D, a long shortcut between the account cluster and the billing cluster. That one link is the "airport."

### Searching the graph

A user asks: *"How do I get my money back?"* The pipeline wants the top `k = 2` chunks with `ef_search = 3`. The question embeds to Q = (7.4, 2.3), right next to the refund chunk.

**Layer 1.** Start at the entry point A: distance 6.53. A's layer-1 neighbour D is at 0.5, closer, so hop to D. D's only layer-1 neighbour is A, which is farther. Layer 1 is done; drop to layer 0 at D.

**Layer 0.** Keep two lists: *results*, the best `ef_search = 3` chunks found so far, and *candidates*, chunks still to explore. Both start with D (0.5).

1. Explore D's neighbours: C (5.41), E (0.92), F (1.75). C and E fill the results. F arrives when the list is full but beats the worst entry (C), so it replaces C. Results: D, E, F.
2. Explore E. Its neighbours D, C and F are all checked already. Nothing changes.
3. Explore F. Its neighbours E and D are checked already.
4. The next candidate is C, at 5.41. That's farther than the worst result (F, 1.75), so nothing reachable through C can improve the results. **Stop.**

Return the top 2: **D (Refund policy, 0.5)** and **E (Cancel subscription, 0.92)**.

The search made 5 distance calculations, and chunk B was never compared at all. At six chunks that saving is trivial. At millions, a search typically touches a few hundred or a few thousand vectors, which is why it takes milliseconds.

## How layers are assigned

Nobody chooses the layers, and `M` doesn't directly assign them either. When a vector is inserted, the database draws its top layer at random:

```
level = floor( -ln(U) × mL )      where U ~ Uniform(0, 1] and mL = 1 / ln(M)
```

The node then appears on every layer from 0 up to `level`.

For `M = 2`, `mL ≈ 1.44`. A draw of U = 0.8 gives `-ln(0.8) × 1.44 ≈ 0.32`, which rounds down to level 0. A draw of U = 0.3 gives `≈ 1.74`, so level 1. In the example, A and D got lucky draws like the second.

This formula makes the probability of reaching level `l` or higher equal to `1 / M^l`. With the common `M = 16`, about 1 in 16 nodes reaches layer 1, 1 in 256 reaches layer 2, and so on. The number of layers grows like log base M of your collection size: roughly 5 for a million chunks.

Two design choices are hiding here:

- **Random, not content-based.** A refund chunk is as likely to reach the top as a password chunk. The database doesn't need to analyse your data, it works the same for bulk loads and one-at-a-time inserts, and shortcuts end up spread evenly across every topic.
- **Tied to M.** Each node has about M links, and each layer up has about M times fewer nodes, so one hop on a higher layer covers roughly the ground of M hops below. That balance keeps the hops per layer small and is why search cost grows close to logarithmically with corpus size.

Most databases don't expose `mL`; they derive it from `M`.

## The stopping rule and ef_search

On the upper layers, the search tracks one best node and stops when no neighbour is closer. On layer 0, it runs the loop from the worked example:

```
while candidates is not empty:
    c = nearest unexplored candidate
    if distance(c) > distance(worst in results):
        stop                      # nothing left can improve the results
    for each neighbour n of c not yet visited:
        if results has room or distance(n) < distance(worst in results):
            add n to candidates and results
            trim results to ef_search
return the top k of results
```

This answers a question that comes up a lot: *what if the search starts right on the best chunk?* It can't know that. "Best" only means something relative to the neighbours, so it still checks them. The search just ends quickly, because nothing beats where it started.

`ef_search` sets the size of the results list. With `ef_search = 1`, the search stops at the first chunk with no closer neighbour, which is fast but can get stuck on a chunk that's only locally best. A larger value makes the "worst result" bar farther away, so the search explores more of the neighbourhood before stopping. It must be at least k.

## The three knobs

| Setting | Set when | Raising it | Typical starting point |
|---|---|---|---|
| `M` | Index build | More links per node: better recall, more memory, slower build | 12–16 |
| `ef_construction` | Index build | More thorough search during inserts: better graph, slower build | 64–200 |
| `ef_search` | Every query | More candidates kept: better recall, slower queries | 40, and at least k |

For RAG, `ef_search` is the cheapest experiment. If answers are poor even though the right content is in your corpus, try raising it before you rework your chunking or prompts. It changes nothing about the index and can be set per query.

## Running it

The code lives in [rag-scratch-to-speciality](https://github.com/sesha-2k3/rag-scratch-to-speciality/tree/main/code/indexing) on GitHub, in two versions: `hnsw_scratch.py` uses only the Python standard library, and `hnsw_lib.py` does the same with real libraries. The project uses [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/sesha-2k3/rag-scratch-to-speciality.git
cd rag-scratch-to-speciality

uv sync
uv run python code/indexing/hnsw_scratch.py
```

The scratch version builds the exact graph from the worked example, checks the layer formula over a million draws, and measures recall against brute force. Expected output:

```
Finished graph
  layer 1  A -> D
  layer 1  D -> A
  layer 0  A -> B, C
  layer 0  B -> A, C, D
  layer 0  C -> B, A, D
  layer 0  D -> E, F, C
  layer 0  E -> D, C, F
  layer 0  F -> E, D
  entry point: A

Top 2 for 'How do I get my money back?'
  D  Refund policy          distance 0.50
  E  Cancel subscription    distance 0.92
  distance calculations: 5 of 6

Levels drawn for 1,000,000 nodes with M = 16
  level >= 0: 1,000,000  (expected ~1,000,000)
  level >= 1:    62,425  (expected ~62,500)
  level >= 2:     3,899  (expected ~3,906)
  level >= 3:       253  (expected ~244)
  level >= 4:        22  (expected ~15)

Recall@10 on 3000 random 32-d vectors (50 queries)
  ef_search=10   recall 0.49   avg distance calls    180 of 3000
  ef_search=20   recall 0.66   avg distance calls    281 of 3000
  ef_search=50   recall 0.88   avg distance calls    524 of 3000
  ef_search=100  recall 0.95   avg distance calls    855 of 3000
```

The last table is the whole speed-for-accuracy tradeoff in four lines: at `ef_search = 100`, the search finds 95% of the true top 10 while computing under a third of the distances. Uniform random vectors are close to a worst case for ANN search; real embeddings have structure that makes recall higher at the same cost. The scratch version also simplifies two things real libraries do: it picks neighbours purely by distance, while libraries use a heuristic that prefers neighbours in different directions, and it has no deletion support.

### The library version

In practice you'll use a library or a database. The core of `hnsw_lib.py`, with [hnswlib](https://github.com/nmslib/hnswlib) and real embeddings:

```python
import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer

texts = [
    "Reset your password",
    "Change account email",
    "Set up two-factor login",
    "Refund policy",
    "Cancel subscription",
    "Update billing card",
]

model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(texts, normalize_embeddings=True)

index = hnswlib.Index(space="cosine", dim=vectors.shape[1])
index.init_index(max_elements=len(texts), M=16, ef_construction=200, random_seed=42)
index.add_items(vectors, np.arange(len(texts)))
index.set_ef(50)

query = model.encode(["How do I get my money back?"], normalize_embeddings=True)
labels, distances = index.knn_query(query, k=2)
```

Run it with:

```bash
uv sync --group lib
uv run --group lib python code/indexing/hnsw_lib.py
```

Exact scores depend on the model version, so I don't list expected output for this one. With hnswlib, `space="cosine"` returns `1 - cosine similarity`, and `space="l2"` returns squared Euclidean distance, not the plain distance.

If your vectors live in Postgres, pgvector builds the same structure with one statement:

```sql
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

SET hnsw.ef_search = 40;
```

## HNSW in production

The algorithm above is the easy part. These are the issues that show up after launch.

### Deletions slowly damage the graph

Some nodes act as bridges between regions of the graph. Remove one, along with its links, and the nodes behind it may become unreachable. They're still in your database, but no search path leads to them, and queries whose answers live there quietly retrieve something else. No error is raised.

One deletion rarely does visible harm because real graphs have redundant links. The damage builds up, and it matters most in RAG systems with changing documents, because updating a document usually means deleting its old chunks and inserting new ones.

Implementations handle this differently. Some mark deleted nodes as *tombstones*: the node stays in the graph so searches can pass through it, but it's excluded from results, and a periodic cleanup rebuilds the affected areas. Others delete immediately and try to reconnect the neighbours. Find out which your database does, because this varies more than any other part of HNSW.

### Filters can starve your results

RAG queries often filter by metadata: only this tenant's documents, only the API docs. If the database searches first and filters afterward, you can ask for 5 chunks and get 1, because the nearest neighbours mostly belonged to other tenants. Fixes include raising `ef_search` for filtered queries, partial indexes per document type (in pgvector, an index with a `WHERE` clause), database features that filter during the graph walk (recent pgvector versions add iterative index scans for this), or separate collections per tenant.

### Recall can drift without anyone noticing

There's no built-in alarm for a degrading index. The practical fix is a small retrieval test set: a handful of questions where you know which chunks should come back. Rerun it on a schedule, and treat a drop in recall as the signal to reindex, rather than waiting for users to notice missing answers.

### Memory and build time

HNSW keeps the graph in memory for speed. The links are small next to the vectors themselves (a 768-dimensional float32 vector is about 3 KB), but at hundreds of millions of chunks, both add up. Building is the slow side of the tradeoff: every insert runs a search, so a large initial build can take far longer than any query. Run it as part of ingestion, not on the request path.

## When HNSW isn't the right choice

HNSW is the default in most vector databases for good reason: fast queries, high recall, and good behaviour with high-dimensional embeddings. Consider alternatives when:

- **Your corpus is small** (tens of thousands of chunks). Brute force is exact, simple, and often fast enough.
- **You rebuild constantly at large scale, or delete heavily.** IVF-style indexes, which group vectors into clusters and search only the nearest clusters, build much faster and cope better with deletions, at some cost in query speed or recall.
- **Memory is the binding constraint.** Quantized indexes compress vectors to fit far more into RAM.

## Key takeaways

- HNSW is built from your embeddings after the embedding step, and it speeds up semantic similarity search rather than replacing it.
- It's a stack of graphs: sparse layers with long links on top, every chunk on the bottom. Search goes top-down, hopping toward the query.
- Layers are assigned randomly, with each layer about M times smaller than the one below.
- The search stops when the nearest unexplored candidate is worse than the worst result kept. `ef_search` controls how much it explores before that happens.
- It's approximate. Measure recall, tune `ef_search` first, and plan for deletions and filtering from the start.

## Further reading

- Yu. A. Malkov and D. A. Yashunin, [Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320), the original HNSW paper.
- Kodesage, [Vector Indexes: HNSW vs IVFFLAT vs IVF_RaBitQ](https://kodesage.ai/blog/vector-indexes-hnsw-vs-ivfflat-vs-ivf-rabitq), a practical comparison of index types, including deletion behaviour and filtering.
