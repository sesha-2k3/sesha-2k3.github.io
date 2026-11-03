---
title: "What an Agent Looks Like to a GPU"
pubDatetime: 2026-11-02T00:00:00Z
featured: true
draft: false
kind: notes
category: AI
tags:
  - agents
  - llm
  - inference
description: "The GPU has never heard of your agent. It sees separate requests that grow every turn, with idle gaps between them. Measured on a MacBook: the same 10-turn agent in 31 seconds or 78, depending on one line of the prompt."
---

## Same agent, 31 seconds or 78

I ran the same 10-turn coding agent twice on my MacBook Pro, against `llama3.2:1b` in Ollama. Same model, same tools, same prompts, same settings. The only difference was one line at the top of the system prompt in the second run: the current time.

```
Default run:              Wall time: 31.11s    Prefill time across all turns:  8.46s
Timestamp in the prompt:  Wall time: 78.02s    Prefill time across all turns: 54.69s
```

The second run spent six and a half times longer just reading its prompts. By turn 10, it was waiting 8.7 seconds before producing its first token, where the default run waited 0.35 seconds.

Nothing about the agent's logic changed. What changed is how the request looked to the inference server, and to explain that, you have to stop looking at your agent the way your framework shows it to you and start looking at it the way the GPU does. That's what this post is about.

## Your agent is a loop. The GPU sees requests.

Strip any agent framework down and you find something like this:

```python
messages = [system_prompt, user_task]
while True:
    reply = llm.chat(messages)          # one HTTP request, with the WHOLE list
    messages.append(reply)
    if reply.is_final:
        break
    result = run_tool(reply.tool_call)  # runs on your machine; the GPU waits
    messages.append(result)
```

Three things in those seven lines matter more than they look:

1. **Every call to `llm.chat` is a separate request.** The server doesn't know these requests belong to one task. "Agent session" exists in your code, not on the GPU.
2. **The whole `messages` list goes over the wire every time.** LLM APIs are stateless, so turn 10 resends the system prompt, the tool definitions, and every earlier reply and tool result. The context grows every turn.
3. **`run_tool` runs between requests.** While a test suite runs or a file is read, there's no request in flight for this session. The GPU has nothing to do for it.

If you've used a chat app, you've already seen this from the other side. While you read a reply and type your next message, the server isn't working on your conversation. When you hit send, the whole conversation goes back to the server as one new request. An agent works the same way, except the "reader" is your code: it parses a tool call, runs the tool, appends the result, and sends the next request immediately, with no human in between. That makes the gaps short and frequent (a fifth of a second to read a file, a few seconds to run tests) instead of long and rare.

## What happens inside one request

Each request goes through two phases on the GPU, and they behave very differently.

**Prefill** reads the prompt. The model processes all the prompt tokens at once, in parallel, and builds the KV cache: the per-token attention state it needs in order to keep generating. Because thousands of tokens go through the model in a few large matrix multiplications, prefill is fast per token. Time-to-first-token is mostly prefill.

**Decode** writes the reply, one token at a time. Each step has to read all the model's weights and the whole KV cache from memory just to produce one token, so decode is slow per token.

My MacBook's numbers from the run with the timestamp, turn 1:

```
turn    prompt  out  prefill  decode
   1      2035   60   1.314s  0.843s
```

The model read 2,035 tokens in 1.3 seconds and wrote 60 tokens in 0.8 seconds. That's about 1,550 tokens per second for prefill and 71 for decode. **Per token, writing cost roughly 20 times more than reading.**

That ratio explains a lot about agents. An agent's turns are read-heavy: thousands of tokens of context, a few dozen tokens of tool call. Most of the per-turn time is spent reading, and the amount to read keeps growing.

## The context grows every turn

Here's the full run with the timestamp at the top of the prompt. `prompt` is the number of tokens sent in each request:

```
turn    prompt  out  prefill  decode  other  tool  tool
   1      2035   60   1.314s  0.843s  0.00s  0.3s  list_files
   2      2497   60   1.644s  0.846s  0.03s  0.2s  read_file
   3      4459   60   3.164s  0.888s  0.03s  4.0s  run_tests
   4      5746   60   4.283s  0.910s  0.04s  0.2s  read_file
   5      7437   60   5.900s  0.942s  0.08s  0.1s  edit_file
   6      7644   60   6.136s  0.950s  0.10s  4.0s  run_tests
   7      8661   60   7.179s  0.963s  0.09s  0.4s  grep
   8      9383   60   8.103s  0.982s  0.12s  0.1s  edit_file
   9      9590   60   8.271s  0.981s  0.11s  4.0s  run_tests
  10      9932   60   8.695s  0.993s  0.12s  0.0s  (final)

Wall time:             78.02s
GPU busy (P + D):      63.99s
Waiting on tools:      13.30s
Busy fraction:         82.0%
Prompt tokens sent across all turns: 67,384
Prefill time across all turns:      54.69s
```

The conversation ends at about 10,000 tokens. But because each turn resends everything before it, the server read **67,384** prompt tokens to get there, 6.7 times the size of the final conversation.

You can work out why with a little arithmetic. If the fixed prefix (system prompt plus tools) is `S` tokens and each turn adds about `a` tokens, turn `k` sends `S + a(k − 1)` tokens. Adding that up over `n` turns gives `nS + a·n(n − 1)/2`. The second term grows with the square of the number of turns. Doubling the length of an agent task roughly quadruples the prefill work it causes.

Two more details in the table are worth a second look:

- **Prefill throughput falls as the context grows.** Turn 1 read 1,549 tokens per second; turn 10 read 1,142. Attention compares every token with every earlier token, so each token costs more in a longer context.
- **Decode slows down too.** The 60-token reply took 0.84 seconds at turn 1 and 0.99 seconds at turn 10, a drop from 71 to 60 tokens per second. Each decode step reads the entire KV cache, and the cache keeps growing.

So a long agent doesn't just do more work. Each unit of work also gets more expensive.

## The gaps

Look at the `tool` column. The fake `run_tests` takes 4 seconds, three times, and the other tools take a fraction of a second. In the default run, the agent spent 13.3 of its 31 seconds waiting on tools. For that time, the GPU had no work from this session.

On a laptop, the GPU simply sits idle. On a shared inference server, it doesn't: the scheduler fills the gaps with other users' requests. But the server still has to decide what to do with this session's KV cache while the tool runs. It can:

- **Keep it in GPU memory**, so the next turn doesn't have to recompute it. That memory is then unavailable to every other session for as long as the tool runs.
- **Throw it away**, freeing the memory, and recompute it all when the next request arrives.
- **Move it** to CPU memory or disk and bring it back later, which costs transfer time.

None of these is free, and the right choice depends on how big the cache is and how long the gap lasts. Gaps in an agent are short and unpredictable. How big that cache actually is, and why that makes this decision so important, is a topic for a future post.

## Why the default run was fast

Here's the default run, without the timestamp:

```
turn    prompt  out  prefill  decode  other  tool  tool
   1      2023   45   1.409s  0.673s  0.00s  0.3s  list_files
   2      2469   59   0.309s  0.862s  0.01s  0.2s  read_file
   3      4429   60   1.523s  0.915s  0.01s  4.0s  run_tests
   4      5716   60   1.100s  0.908s  0.01s  0.2s  read_file
   5      7408   60   1.606s  0.948s  0.02s  0.1s  edit_file
   6      7615   60   0.174s  0.956s  0.02s  4.0s  run_tests
   7      8632   60   1.026s  0.967s  0.02s  0.4s  grep
   8      9354   60   0.757s  0.983s  0.02s  0.1s  edit_file
   9      9561   60   0.201s  0.991s  0.02s  4.0s  run_tests
  10      9903   60   0.353s  0.990s  0.02s  0.0s  (final)

Wall time:             31.11s
GPU busy (P + D):      17.65s
Waiting on tools:      13.30s
Busy fraction:         56.7%
Prompt tokens sent across all turns: 67,110
Prefill time across all turns:       8.46s
```

The `prompt` column barely differs from the other run: about 67,000 tokens sent in total either way. Yet prefill took 8.5 seconds instead of 54.7, and it doesn't grow turn by turn. It jumps around.

![Prefill time per turn for the two runs. With the prefix reused, prefill stays under 2 seconds and tracks how much new content each turn added. With a timestamp at the top of the prompt, it climbs steadily to 8.7 seconds.](/images/agents-gpu/real-prefill.png)

The pattern makes sense once you compare prefill time with the number of tokens each turn *added*. Turn 3 added a 1,900-token test log and took 1.5 seconds. Turn 6 added a 150-token edit confirmation and took 0.17 seconds. Across all turns, the default run spent roughly a millisecond per new token, and almost nothing on tokens it had seen before.

That's because Ollama keeps the KV cache from the previous request and compares the new prompt with it from the start. Whatever matches exactly is reused, and only the rest is prefilled. In an agent, the new prompt is the old one plus a reply and a tool result, so almost everything matches.

Now the slow run makes sense. The timestamp sits at the very top of the system prompt and changes every turn, so the match fails at almost the first token, and the server recomputes everything. One line broke the reuse for all 10,000 tokens behind it. This is called prefix caching, and plenty of common agent habits break it just as easily as a timestamp does. I'll cover those in a future post.

Two lessons from this comparison:

- **Ollama's prompt token count didn't show the difference.** Its `prompt_eval_count` reported the full prompt length in both runs, whether or not the cache was reused. I expected it to drop in the default run, and it didn't. If you want to know whether caching is working, measure prefill time, not token counts.
- **A busier GPU isn't a better agent.** The slow run had an 82% busy fraction; the fast one, 57%. The extra "utilization" was the GPU redoing work it had already done.

## The same agent at datacenter scale

A 1B model on a laptop is a small setup. To see the shape at a more typical scale, I wrote a deterministic simulation of a 10-turn agent using rough numbers for an 8B model on a datacenter GPU: 8,000 prefill tokens per second and 60 decode tokens per second. It models the same pattern (a 3,000-token prefix, tool results from 150 to 1,800 tokens, a few slow test runs) with and without prefix reuse:

```
                     Cache off   Cache on
Tokens prefilled        77,730     10,100
Prefill time             9.72s      1.26s
Decode time             19.17s     19.17s
Tool (idle) time        20.00s     20.00s
Wall time               48.88s     40.43s
```

![Gantt chart of the simulated 10-turn agent. Each turn is its own row; red prefill bars widen every turn, blue decode bars vary with output length, and dashed gray tool gaps take up 20 of the 49 seconds.](/images/agents-gpu/sim-timeline.png)

Reading down the chart, the red prefill bar gets wider every turn even though each turn adds only a few hundred to a couple of thousand tokens of new information. And the dashed gaps are long: the GPU has no work from this session for 20 of the 49 seconds.

The table also shows what caching can't fix. It cut prefill by 87%, but decode and tool time stayed exactly the same, so the session got only 17% faster. On a faster GPU, prefill is a smaller part of the session than on a laptop, and the idle gaps become a larger part of it.

## Run it yourself

From the root of the [REPO](https://github.com/sesha-2k3/rag-scratch-to-speciality/tree/v1) repo:

```bash
uv sync
uv run python code/agents_gpu/agent_gpu_sim.py
uv run python code/agents_gpu/agent_gpu_sim.py --cache
uv run python code/agents_gpu/agent_gpu_sim.py --plot sim.png
```

The simulation is deterministic, so your output should match the table above exactly.

The real measurement needs [Ollama](https://ollama.com) and a small model:

```bash
brew install ollama
ollama serve                 # in another terminal, unless the Ollama app is running
ollama pull llama3.2:1b
python3 agent_gpu_ollama.py
python3 agent_gpu_ollama.py --bust-cache --plot real.png
```

Full code: [`agent_gpu_sim.py`](https://github.com/sesha-2k3/rag-scratch-to-speciality/blob/v1/code/agents_gpu/agent_gpu_sim.py) · [`agent_gpu_ollama.py`](https://github.com/sesha-2k3/rag-scratch-to-speciality/blob/v1/code/agents_gpu/agent_gpu_ollama.py)
The tools in this script are fake (`time.sleep` and canned text), so every run sends the same prompts and you're measuring only the serving side. Your timings will differ from mine depending on your Mac, but the shape should not: flat-ish prefill in the default run, steadily climbing prefill with `--bust-cache`.

## Takeaways

- The GPU has never heard of your agent. It sees separate, stateless requests that share most of their text.
- Every turn resends the entire conversation, so total prefill work grows with the square of the number of turns. My 10-turn run read 67K prompt tokens to build a 10K-token conversation.
- Prefill reads many tokens cheaply; decode writes one token at a time and costs about 20 times more per token. Both get slower as the context grows.
- Tool calls leave gaps where the GPU has no work from your session, and the server has to decide whether to keep your KV cache, throw it away, or move it.
- Reusing an exactly matching prefix avoids most of the re-reading, and one changing line at the top of your prompt is enough to break it. Check this by measuring prefill time, not token counts.