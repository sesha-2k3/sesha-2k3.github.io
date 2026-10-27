---
title: "RAG from Scratch, Part 4: Evaluating Retrieval"
pubDatetime: 2026-10-26T00:00:00Z
featured: true
draft: false
kind: notes
category: AI
tags:
  - rag-series
  - rag
  - llm
  - evaluation
description: "Precision, recall, MRR, and nDCG computed by hand and in code, the nDCG mistake that hides missed chunks, and whether 82% vs 78% accuracy is a real difference."
---

**RAG from Scratch series:** [Part 0: Overview](/posts/rag-00-overview/) · [Part 1: Chunking](/posts/rag-01-chunking/) · [Part 2: Embeddings, tokens, and metadata](/posts/rag-02-embeddings/) · [Part 3: The query-time flow](/posts/rag-03-query-flow/) · **Part 4: Evaluating retrieval** · [Part 5: BM25](/posts/rag-05-bm25/)

## Is 82% better than 78%?

You change your chunking strategy and rerun your test set of 50 questions. The old pipeline answered 39 correctly (78%). The new one answers 41 (82%). Ship it?

Not yet. By the end of this post, we'll show that this 4-point gap is entirely consistent with random noise. Getting there means answering three questions first: what exactly to measure, how to measure it correctly, and what to test it on.

## Evaluate retrieval and generation separately

A wrong answer from a RAG system has two possible causes:

- **Retrieval failure:** the right chunk never reached the LLM. No prompt change can fix this.
- **Generation failure:** the chunk was retrieved, but the LLM ignored it, misread it, or trusted a conflicting chunk instead.

If you only grade final answers, you learn *that* the system failed but not *where*. So we'll measure retrieval on its own first, then generation.

## Retrieval metrics: precision, recall, hit rate

Throughout this section, one query has three relevant chunks, `{A, B, C}`, and the retriever returns five: `[X, A, Y, B, Z]`.

| Metric | Question it answers | Value |
|---|---|---|
| Precision@5 | What fraction of retrieved chunks are relevant? | 2/5 = 0.40 |
| Recall@5 | What fraction of relevant chunks were retrieved? | 2/3 = 0.67 |
| Hit rate@5 | Was at least one relevant chunk retrieved? | 1 |
| Recall@1 | Is the top result relevant? | 0 |

Each one is a line or two of Python:

```python
def precision_at_k(retrieved, relevant, k):
    return sum(d in relevant for d in retrieved[:k]) / k

def recall_at_k(retrieved, relevant, k):
    return sum(d in relevant for d in retrieved[:k]) / len(relevant)

def hit_rate_at_k(retrieved, relevant, k):
    return float(any(d in relevant for d in retrieved[:k]))
```

Two rules make these numbers meaningful:

**Use the k you actually send to the LLM.** If your pipeline passes the top 3 chunks, report @3. Metrics at @10 will flatter you with chunks the LLM never sees.

**Recall matters most.** A relevant chunk that retrieval misses can't be recovered at any later stage.

Comparing recall at two values of k is also a useful diagnostic. If recall@10 is 0.95 but recall@3 is 0.60, retrieval is *finding* the right chunks but *ranking* them poorly. The fix is a reranker, which Part 6 covers, not a new embedding model.

## MRR: how high is the first useful result?

Mean Reciprocal Rank asks one question: on average, how close to the top is the first relevant result? For each query, find the rank of the first relevant chunk, take 1 divided by that rank (or 0 if there isn't one), and average across queries.

```python
def reciprocal_rank(retrieved, relevant, k=10):
    for rank, doc in enumerate(retrieved[:k], start=1):
        if doc in relevant:
            return 1 / rank  # only the FIRST relevant hit counts
    return 0.0
```

Over four queries:

| Query results | First relevant | Reciprocal rank |
|---|---|---|
| `[R, X, X, X, X]` | rank 1 | 1.000 |
| `[X, X, R, X, X]` | rank 3 | 0.333 |
| `[X, R, R, X, X]` | rank 2 | 0.500 |
| `[X, X, X, X, X]` | none | 0 |

MRR = (1 + 0.333 + 0.5 + 0) ÷ 4 = **0.46**.

The scores drop steeply: rank 1 is worth 1.0, rank 2 is worth 0.5, and rank 10 is worth only 0.1. That suits factoid questions, where one chunk holds the answer and it needs to be near the top.

MRR also has blind spots. It ignores everything after the first hit, so `[R, X, X, X, X]` and `[R, R, R, R, R]` both score 1.0. It's also very sensitive to incomplete labels: if a correct chunk at rank 1 was never labeled as relevant, a query that should score 1.0 might score 0.5. And leave unanswerable questions out of MRR, since they have no relevant chunk to rank.

## nDCG: how good is the whole ordering?

Normalized Discounted Cumulative Gain scores the entire ranking from 0 to 1. Each part of the name is one step:

- **Gain:** each result's relevance, either 0/1 or graded such as 0/1/2.
- **Cumulative:** add the gains up.
- **Discounted:** divide each gain by log₂(rank + 1), so lower ranks count less.
- **Normalized:** divide by the best possible score, the *ideal* DCG.

The discounts fall gently compared with MRR:

| Rank | 1 | 2 | 3 | 4 | 5 | 10 |
|---|---|---|---|---|---|---|
| Discount | 1.000 | 0.631 | 0.500 | 0.431 | 0.387 | 0.289 |

For `[X, A, Y, B, Z]`, the relevant chunks sit at ranks 2 and 4, so DCG = 0.631 + 0.431 = 1.062.

### The mistake that hides missed chunks

To normalize, you need the ideal DCG: the score of a perfect ranking. Here's the correct way to build it:

1. List **every relevant chunk in your labels**, including ones retrieval missed.
2. Sort them by relevance, highest first.
3. Place them at ranks 1, 2, 3, and so on, and fill the rest with zeros up to k.
4. Compute DCG on that list.

The ideal ranking here is `[A, B, C, -, -]`, so ideal DCG = 1 + 0.631 + 0.5 = 2.131, and **nDCG@5 = 1.062 ÷ 2.131 = 0.50**.

The common mistake is to build the ideal from only the relevant chunks that were *retrieved*, A and B. That gives an ideal of 1 + 0.631 = 1.631 and an nDCG of 0.65. It looks better, but only because it has stopped penalizing the system for missing C entirely.

The code avoids this by building the ideal from the labels, not the results:

```python
import math

def dcg(gains):
    # rank 1 -> divide by log2(2)=1, rank 2 -> log2(3), ...
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))

def ndcg_at_k(retrieved, relevance, k):
    """relevance: {chunk_id: grade} from your ground-truth LABELS."""
    actual = [relevance.get(doc, 0) for doc in retrieved[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]  # from labels, not results
    idcg = dcg(ideal)
    return dcg(actual) / idcg if idcg > 0 else 0.0
```

Running both versions:

```
nDCG@5      = 0.498
nDCG@5 with the WRONG ideal (forgets C) = 0.65
```

Queries with no relevant chunks have an ideal DCG of 0, so nDCG is undefined for them. Skip those queries rather than counting them as 0.

### Same recall, different nDCG

Recall can't tell these three rankings apart, but nDCG can:

```
['A', 'B', 'X', 'Y', 'Z']: recall@5=0.67 nDCG@5=0.77
['X', 'A', 'Y', 'B', 'Z']: recall@5=0.67 nDCG@5=0.50
['X', 'Y', 'Z', 'A', 'B']: recall@5=0.67 nDCG@5=0.38
```

This is why nDCG is the metric for rerankers. A reranker reorders the same candidates, so recall over the candidate set can't change. nDCG shows whether it moved the right chunks up.

### Graded relevance

Some chunks answer the question directly while others are only side notes. nDCG can reflect that with graded labels. Suppose A and C are direct answers (grade 2), B and D are supporting context (grade 1), and the retriever returns `[B, X, A, Y, C]`:

- The ideal ranking is `[2, 2, 1, 1, 0]`, with a DCG of 4.193.
- The actual DCG is 1 + 2(0.5) + 2(0.387) = 2.774.
- **nDCG@5 = 0.66.**

Recall and binary MRR can't see that a direct answer should outrank a side note.

### Why normalize at all?

Without normalization, queries with more relevant chunks get higher scores automatically. A query with one relevant chunk retrieved at rank 1 has a DCG of 1.0. A query with five relevant chunks retrieved at ranks 2 through 5 has a DCG of 1.949, which is higher even though the ranking is worse. After normalizing, they score 1.00 and 0.66, which matches reality.

### Which metric when

| Metric | Measures | Best for |
|---|---|---|
| Recall@k | Did retrieval find all relevant chunks? | Questions that need several chunks |
| MRR | Rank of the first relevant chunk | Factoid questions with one answer chunk |
| nDCG@k | Quality of the whole ordering | Multiple or graded relevant chunks, rerankers |

## Building a test set

Metrics are only as good as the questions behind them. There are three common sources, from most realistic to fastest:

1. **Real user questions** with hand-labeled relevant chunks.
2. **Questions written by domain experts.**
3. **LLM-generated questions** from sampled chunks.

Generated questions have a trap: they tend to copy the chunk's wording. A chunk about vacation policy produces "How many days of paid time off do employees accrue per month?", while a real user types "how much vacation do I get?" Retrieval looks much better on the first.

To reduce this, instruct the generator to paraphrase, generate questions that need several chunks, include messy and ambiguous questions, and check that your synthetic set ranks pipelines in the same order as 20–30 hand-labeled real questions.

### Include questions you can't answer

A good test set also contains questions your documents *can't* answer:

- **Out-of-scope questions:** nothing in the documents covers them.
- **Near misses:** contractor leave, when the handbook only covers employees.
- **False premises:** "Why was Atlas cancelled?" when it wasn't.

Track two numbers: the abstention rate on unanswerable questions, which you want high, and the false abstention rate on answerable ones, which you want low. Tightening the similarity threshold from Part 3 raises both. For HR, legal, or medical content, accepting more false abstentions is usually the right trade.

## Generation metrics

Once retrieval is measured, the remaining question is what the LLM did with the chunks. The RAGAS framework uses four metrics, two per stage:

| Metric | Question | Stage |
|---|---|---|
| Context recall | Did retrieval find everything needed? | Retrieval |
| Context precision | Is the retrieved context relevant and ranked high? | Retrieval |
| Faithfulness | Is every claim in the answer supported by the context? | Generation |
| Answer relevance | Does the answer address the question? | Generation |

Faithfulness catches a failure that answer correctness misses: **correct but unfaithful**. Suppose someone asks for the capital of Australia, retrieval returns chunks about Sydney tourism, and the LLM answers "Canberra" from its training knowledge. The answer is right, but it hides a retrieval bug. On your private documents, which the model has never seen, the same behavior produces confident wrong answers.

### LLM-as-judge

Faithfulness and answer quality are usually graded by another LLM. That works, but LLM judges have known biases:

- **Position bias:** preferring whichever answer comes first.
- **Verbosity bias:** preferring longer answers.
- **Self-preference:** favoring outputs from the same model family.
- **Leniency and drift:** grading generously, or inconsistently over time.
- **Factual blind spots:** missing errors the judge itself doesn't know about.

Each has a countermeasure. Run comparisons in both orders and count only consistent wins. Use specific yes-or-no criteria instead of 1–10 scales. Give the judge the reference answer or the retrieved context. Ask for reasoning before the verdict. Use a judge from a different model family. And calibrate against about 50 human-graded answers before trusting it.

## When better retrieval means worse answers

Here's a result that surprises people. Increasing chunk size from 300 to 1,000 tokens can *raise* recall@5 while making answers *worse*.

Bigger chunks pass the "is the answer somewhere in here?" check more easily, so recall goes up. But the LLM now receives 5,000 tokens of mostly irrelevant text, and each chunk's embedding is blurrier, as Part 1 discussed. The lesson: **measure final answers, not just retrieval.** A common fix, small-to-big retrieval, comes in Part 6.

## Back to 82% vs 78%

Now we can answer the opening question. With 50 questions, 82% vs 78% means 41 correct vs 39. The key is to compare the two pipelines **on the same questions**, and only the questions where they disagree carry information:

- Both correct: 35
- Only A correct: 6
- Only B correct: 4
- Both wrong: 5

If the pipelines were equally good, each of the 10 disagreements would be a coin flip between A and B. McNemar's test checks whether a 6-to-4 split is surprising under that assumption:

```python
chi2 = (a_only - b_only) ** 2 / (a_only + b_only)   # (6 - 4)² / 10 = 0.4
p_chi2 = math.erfc(math.sqrt(chi2 / 2))              # chi-square, 1 degree of freedom

# Exact version: under "equal systems", each disagreement is a coin flip
n, x = a_only + b_only, max(a_only, b_only)
p_exact = min(1.0, 2 * sum(math.comb(n, i) for i in range(x, n + 1)) / 2 ** n)
```

The output:

```
A: 41/50, B: 39/50
Disagreements: A right & B wrong = 6, B right & A wrong = 4
Chi-square = 0.40, p = 0.53
Exact binomial p = 0.75
Not significant: the 4-point gap is consistent with random noise.
```

A p-value of 0.53 means a split at least this lopsided would happen about half the time even if the pipelines were identical. The 4-point gap isn't evidence of anything. With only 10 disagreements, the exact binomial test is the more trustworthy of the two, and at 0.75 it's even less convincing.

Instead of shipping on the number, read the 10 questions where the pipelines disagree, since that's usually where the real insight is. Grow the test set into the hundreds, rerun any non-deterministic steps several times, and if the pipelines are still tied, choose on cost, speed, and simplicity.

## Putting it into practice: switching embedding models

All of this comes together when you change a core component, such as the embedding model:

1. Build the new index alongside the old one, with the same chunking.
2. Compare retrieval with recall@k at your real k, plus MRR or nDCG, on both real and synthetic questions.
3. Compare end-to-end answers, including abstention on unanswerable questions.
4. Inspect per-question regressions, especially on codes, non-English text, and tables.
5. Check speed, cost, vector dimension, token limit, and where your data is processed.
6. Roll out gradually, watching thumbs-down rate, "I don't know" rate, rephrased questions, and citation clicks. Keep the old index for rollback.
7. Record the model name and version in metadata, and save the evaluation results.

The general rule: change one variable, measure on a fixed test set, inspect per-question differences, and roll out with monitoring.

## The same thing with libraries

[ranx](https://github.com/AmenRa/ranx) computes all the retrieval metrics. It takes ground-truth labels as `Qrels` and your system's output as a `Run`, where scores only define the order:

```python
from ranx import Qrels, Run, evaluate

qrels = Qrels({
    "q1": {"A": 1, "B": 1, "C": 1},
    "q2": {"R2": 1},
})
run = Run({
    "q1": {"X": 5, "A": 4, "Y": 3, "B": 2, "Z": 1},   # [X, A, Y, B, Z]
    "q2": {"X1": 3, "X2": 2, "R2": 1},                # first relevant at rank 3
})

metrics = ["precision@5", "recall@5", "hit_rate@5", "mrr", "ndcg@5"]
print(evaluate(qrels, run, metrics, return_mean=False))  # per query
```

The per-query results for `q1` should match the scratch numbers above.

For significance testing, statsmodels implements McNemar's test from the same table of agreements and disagreements:

```python
from statsmodels.stats.contingency_tables import mcnemar

#       B right  B wrong
table = [[35,      6],   # A right
         [4,       5]]   # A wrong

print(mcnemar(table, exact=False, correction=False))  # chi-square version
print(mcnemar(table, exact=True))                     # exact binomial version
```

## Run it yourself

From the root of the [rag-scratch-to-speciality](https://github.com/sesha-2k3/rag-scratch-to-speciality/tree/part-04) repo:

```bash
uv sync
uv run python code/evaluation/retrieval_metrics_scratch.py
uv run python code/evaluation/compare_pipelines_scratch.py
```

For the library versions:

```bash
uv sync --group lib
uv run --group lib python code/evaluation/retrieval_metrics_lib.py
uv run --group lib python code/evaluation/compare_pipelines_lib.py
```

Full code: [`retrieval_metrics_scratch.py`](https://github.com/sesha-2k3/rag-scratch-to-speciality/blob/part-04/code/evaluation/retrieval_metrics_scratch.py) · [`retrieval_metrics_lib.py`](https://github.com/sesha-2k3/rag-scratch-to-speciality/blob/part-04/code/evaluation/retrieval_metrics_lib.py) · [`compare_pipelines_scratch.py`](https://github.com/sesha-2k3/rag-scratch-to-speciality/blob/part-04/code/evaluation/compare_pipelines_scratch.py) · [`compare_pipelines_lib.py`](https://github.com/sesha-2k3/rag-scratch-to-speciality/blob/part-04/code/evaluation/compare_pipelines_lib.py)

## Takeaways

- Measure retrieval and generation separately, so you know which stage failed.
- Report metrics at the k you actually send to the LLM. Recall matters most, MRR suits single-answer questions, and nDCG captures the whole ordering.
- Build nDCG's ideal ranking from your labels, not from what was retrieved, or missed chunks stop counting against you.
- Test sets need realistic phrasing and unanswerable questions. Synthetic questions that copy chunk wording make retrieval look better than it is.
- A few points of accuracy on a small test set is often noise. Compare pipelines on the same questions with McNemar's test, and read the disagreements.

Want to test yourself? The repo's [evaluation quiz](https://github.com/sesha-2k3/rag-scratch-to-speciality/blob/part-04/quizzes/02-evaluation.md) covers this part.

**Next:** [Part 5: BM25](/posts/rag-05-bm25/)
