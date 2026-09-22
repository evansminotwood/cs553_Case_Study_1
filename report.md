# Case Study 1 Report: Blackjack Tutor

## a. Team Members

- E. Minot Wood (emwood@wpi.edu)
- Andrey Sargsyan (asargsyan@wpi.edu)

## b. Product Description

**Blackjack Tutor** [https://huggingface.co/spaces/emwo/CS553_WPI_Fall_2026_Case_Study_1] is a chatbot that offers up guidance to blackjack players. This allows it to serve as a tutor and aid in learning strong hands vs weak hands and what moves are the correct moves to make. It challenges traditional Blackjack strategy guides as most are fact based instead of conversational or the idea behind it.

**Target audience:** Anyone interested in improving their Blackjack game whether for fun or competition

The app exposes two inference paths:

1. **Remote:** a large hosted LLM accessed via `huggingface_hub.InferenceClient`
   (default path).
2. **Local:** a small model running on the same host via a `transformers` pipeline,
   used either by manual override or automatically as a failover.

**Extra credit implemented:** Adaptive LLM Failover. If the remote API call fails due to
a timeout, rate limit or outage, the app automatically tries the request on the local
model, and labels every response with which model produced it.

## c. Models

| | Remote model | Local model |
|---|---|---|
| **Model** | `openai/gpt-oss-20b` | `Qwen/Qwen3-0.6B` |
| **Architecture** | Decoder-only transformer, open-weight, ~20B parameters | Decoder-only transformer, dense, 0.6B parameters |
| **Access** | Hugging Face Inference API | Runs on the Space's own hardware |
| **Purpose here** | Primary, higher-quality responses | Fast, low-cost fallback and manual "local mode" |
| **Training data** | Not publicly fully disclosed by the provider | Multilingual pretraining + instruction-tuning corpus described in Qwen3's model card |

## d. Performance Analysis

| | Remote (`gpt-oss-20b`) | Local (`Qwen3-0.6B`) |
|---|---|---|
| Response latency | 0.3s | Measured locally on a CPU-only dev machine (no GPU): **~25s cold model load**, then **~13s to generate a 400-token response**. On the Space's actual GPU hardware this should be faster once warm, but ZeroGPU allocations add their own cold-start delay. |
| Resource usage | Everything happens on HF's inference infrastructure. | Uses the Space's own CPU/GPU and RAM directly |
| Failure modes observed | Rate limits, timeouts, transient HTTP errors (handled by our automatic failover) | None observed independently of the host Space being available |

## e. Cost-Based Analysis

**Assumptions for a 1,000-user estimate:**

- 1,000 active users × ~10 messages/user/day = **~10,000 messages/day**
- Each message averages ~200 input tokens (system prompt + recent history) and
  ~300 output tokens (the response) → **~2,000,000 input tokens/day,
  ~3,000,000 output tokens/day**

**Remote model (`gpt-oss-20b` via Inference Providers → Groq, $0.075/1M input tokens,
$0.30/1M output tokens):**
- Output: 3M tokens/day × $0.30/1M = **$0.90/day** (~$27/month)
- Input: 2M tokens/day × $0.075/1M = **$0.15/day** (~$4.50/month)
- **Total ≈ $31.50/month** for 1,000 users at this traffic level. HF's free-tier
  credits ($0.10/month) and PRO credits ($2/month) are both far below this, so any
  real deployment at this scale needs pay-as-you-go billing beyond the included
  credits — cheap in absolute terms, but not "free tier."

**Local model (`Qwen3-0.6B` on Space hardware):** billed per GPU-hour, not per token.
- **ZeroGPU** (what the app currently uses): free, but quota-limited per user/org —
  not sized to guarantee 1,000 concurrent-ish users' worth of sustained local traffic.
- **Dedicated Nvidia T4-small**: $0.40/hr → **$288/month** running continuously,
  regardless of how many messages are actually sent.
- **Dedicated Nvidia T4-medium**: $0.60/hr → **~$438/month**.

**Bottom line:** at this traffic volume, the remote model is roughly **9–14x cheaper**
than reserving dedicated GPU hardware 24/7 for local inference, because the per-token
rate is tiny relative to paying for round-the-clock hardware. The remote model's cost
scales with usage (cheap here, but riskier/less predictable at much higher scale or
under rate limits); the local model's cost is a fixed hardware bill that only pays off
at far higher sustained volume than 1,000 users generate — which is exactly why this
product treats local as a failover tier, not the primary path.

## f. Comments and Concerns

- **Security:** The user's Hugging Face OAuth token is only held in-memory for the duration of the request. There is no server-side rate limiting or abuse protection.
- **Scalability:** the local model is bottlenecked by the Space's own hardware so one
  Space can only have so many concurrent requests before queuing or failing. 
- **Reliability / dependence on an external API:** the app depends on the remote
   API being available, we solve this with the automatic failover
  (see extra credit)

## g. LLM Code Review — Established Model (Claude Sonnet 5, Anthropic)

**Complete prompt used:**

> Analyze all of this code, and the overall product. Look for any issues. Then Identify one meaningful improvement related to performance, reliability, scalability, cost, security, or maintainability. Do not change anything, do not install anything. Just look.

**Response (Claude Sonnet 5):**

> Good — `.env` is properly gitignored and never tracked. Now I have the full picture. Here's the analysis.
>
> ## Findings, by severity
>
> **🔴 Reliability/Correctness — `local_generate` is decorated with `@spaces.GPU` but is also the fallback path when the remote API fails ([app.py:79-94](app.py#L79-L94))**
> On HF Spaces, `@spaces.GPU` dynamically allocates a shared ZeroGPU worker per call — that allocation itself can fail or queue under load. If the remote API is down because HF's infrastructure is degraded broadly, the "fallback" may queue/fail for the same underlying reason, defeating the purpose of failover. This isn't caught — an exception from `local_generate` inside the `except` block ([app.py:163](app.py#L163)) is unhandled and will crash the generator, producing a raw Gradio error instead of a graceful message.
>
> **🟡 Reliability — No backoff/circuit breaker on remote failures**
> Every single message re-attempts the remote API from scratch, even seconds after a known failure, paying the full 15s `timeout` penalty on every request during an outage before falling back ([app.py:138](app.py#L138)). Confirmed from prior session analysis (memory) and matches the code as read today — this is unchanged. There's no cached "remote is currently down" state.
>
> **🟡 Performance — Local model loaded eagerly at import time on module load ([app.py:9-14](app.py#L9-L14))**
> `pipeline(...)` runs at import, meaning every process start (including `pytest` imports, though tests stub `transformers`) pays the ~25s CPU load cost / GPU init cost, even for requests that will only ever use the remote path. This also means `device="cuda"` is hardcoded — the app will crash on any environment without a CUDA device (e.g., local CPU dev, or a misconfigured Space), which contradicts the app's own local-fallback story: the fallback itself is not CPU-portable.
>
> **🟡 Maintainability/Test gap — Tests never exercise the real `respond()` decorator path**
> `local_generate` is wrapped in `@spaces.GPU`, but tests stub `spaces.GPU` as an identity function ([tests/test_app.py:60-61](tests/test_app.py#L60-L61)). That's reasonable for CI portability, but it means the GPU-allocation failure mode (the most likely real failure path on a Space) has zero test coverage — the "automatic failover" extra-credit feature is validated only against a mocked `InferenceClient`, not against a local-model failure.
>
> **🟢 Minor — CI installs are duplicated/inconsistent with `requirements.txt`**
> [test-model.yml:28-29](.github/workflows/test-model.yml#L28-L29) manually appends `gradio[oauth]`, `huggingface_hub`, and `pytest` on top of `uv pip install -r requirements.txt`, meaning the actual app dependencies (`gradio`, `huggingface_hub`) aren't pinned in `requirements.txt` at all — only `transformers` and `accelerate` are. This is a lockfile/reproducibility gap: what CI tests against and what `requirements.txt` declares diverge.
>
> **🟢 Minor — Uncommitted `report.md` sitting in the worktree** (per git status) — not a code issue, just noting it's there un-tracked.
>
> ## One meaningful improvement to recommend
>
> **Reliability: add a stateful circuit breaker for the remote→local failover**, sitting on top of the existing try/except in `respond()` ([app.py:131-167](app.py#L131-L167)).
>
> Right now, once the remote API starts failing, *every* user in *every* session still pays the full 15-second `InferenceClient` timeout before falling back to local, for every single message, for as long as the outage lasts. A simple module-level "remote last failed at T" timestamp with a short cooldown (e.g., skip the remote attempt entirely for 60s after a failure, then let one request retry to see if it recovered) would:
>
> - Cut fallback latency from ~15s to near-zero for every request during an outage (**performance/UX**)
> - Reduce redundant load against a possibly-already-struggling remote endpoint (**reliability**, avoids pile-on)
> - Cost nothing extra to implement — a few lines of shared state, no new dependency
>
> This is a better target than the GPU-fallback-can-also-fail issue because it's a small, self-contained, well-scoped fix directly inside the failover logic the extra-credit feature is built around, whereas the `@spaces.GPU` fallback-of-a-fallback issue would require rethinking the local-inference deployment story (CPU vs GPU) — a bigger, riskier change for a case study.
>
> Want me to implement the circuit-breaker change, or leave this as analysis only (per your "just look" instruction)?

## h. LLM Code Review — Weaker/Local Model (`qwen3-coder`, run locally)

The identical prompt from (g) was run against the actual local model used by the app
(`qwen3-coder`). Used Claude code as the harness and ollama to run and serve the model locally.

**Response:**

> Looking at this codebase, I can analyze it thoroughly without making changes or installing anything.
>
> Code Analysis
>
> This is a Gradio-based Blackjack tutoring application that implements an adaptive LLM failover system. The app has two models:
> 1. Local model: Qwen/Qwen3-0.6B (running on GPU)
> 2. Remote model: openai/gpt-oss-20b (via Hugging Face Inference API)
>
> Key Features
>
> The application implements automatic failover - when the remote API fails for any reason, it seamlessly switches to the local model with no user intervention required.
>
> Issues Identified
>
> 1. Missing Security Controls
>
> The most critical issue is that the app uses a hardcoded Hugging Face token from gr.OAuthToken without implementing proper token validation or security checks. If this were deployed publicly, it could expose users' HF tokens.
>
> 2. Inefficient Error Handling in Streaming
>
> The streaming logic in respond() function has a potential performance issue:
> - The code yields partial responses as they arrive (yield f"🌐 **[Remote API — {REMOTE_MODEL}]**\n\n{response}")
> - But the response variable accumulates tokens, which could lead to memory issues with long responses
> - If the remote fails and falls back to local, it's inefficient to have already streamed partial content
>
> 3. Missing Input Validation
>
> The system doesn't validate:
> - User input length limits
> - System message parameters
> - Token limits for both models
> - Model parameter ranges (temperature, top_p)
>
> 4. No Rate Limiting or Throttling
>
> The application doesn't implement any rate limiting on user requests, which could be problematic if many users access it simultaneously.
>
> 5. Lack of Context Management
>
> The chat history isn't limited in size, so long conversations could consume excessive memory and cause performance issues.
>
> One Meaningful Improvement
>
> Performance & Reliability: The application could benefit from implementing response caching with TTL (Time To Live) for common Blackjack scenarios.
>
> The app is designed as a tutoring tool for basic strategy decisions. Many user queries will be repetitive (e.g., "16 vs dealer 10, what do I do?"). Currently, every query triggers a model inference, even when similar queries have been answered recently.
>
> Implementation: Add an in-memory cache that stores responses to common Blackjack hand scenarios with TTL of 5-10 minutes. This would significantly reduce redundant model calls, improve response times for repeated queries, and decrease both computational costs and API usage.
>
> This is a meaningful improvement because:
> - Performance: Reduces latency for repeated queries
> - Cost: Decreases model inference costs
> - Scalability: Reduces load on both local and remote models
> - Reliability: Improves response consistency under load
>
> The cache would be particularly effective for the fixed Blackjack scenarios that users ask about, making the system more efficient while maintaining the quality of responses.

## i. Human Comparison

Guiding questions to answer here (2 points, must be written by a human):
- Which response was more useful, accurate, and actionable?
  - I believe that claudes response was much more useful, it contained more in-depth details and was overall more accurate as a result of quoting specific files and lines. It also gave a clearer launching point to fix the code.

- Which specific suggestion(s) would you actually implement, and why?
  - I would definitely add the stateful circuit breaker for the remote to local failover from claude, and the local cache from the qwen coder model which would save tokens and  server materials.


## j. Deployment Recommendation

Guiding questions to answer here (2 points, must be written by a human):
- If deploying this for real users, would you choose the remote model, the local model,
  or keep the failover hybrid? Why?
  - I would totally keep the failover hybrid. being able to keep this up consistently for users is super important as depending on how this is used, it could receive several messages in a game. I think depending on the number of users it could also be super helpful to look into the other options for the local models from HF as that could ensure that everyone would have consistent use.
- Would your answer change if cost were the top priority? If privacy were the
  top priority? If reliability were the top priority?
  - I think so. For just above $30 I would probably stick to the free plan to run the local models, which would lose out on some consistency. I can't be certain if this app would be used since Blackjack is a mathmatically solved game, many people might just reach for those harder to understand cheat cards. I'm sure if you can understand it, it would be much better than potentially paying for this service. 
