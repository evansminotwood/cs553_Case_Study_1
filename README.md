---
title: DS553 Fall26
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
  - inference-api
license: mit
---

A Blackjack basic-strategy tutor chatbot built with [Gradio](https://gradio.app), [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/v0.22.2/en/index), and the [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index). Ask it about any hand and it explains the mathematically optimal play — hit, stand, double down, split, or surrender — and why.

## Adaptive LLM Failover (Extra Credit)

By default the app tries the remotely hosted model (`openai/gpt-oss-20b` via `InferenceClient`) first. If that call raises any exception — a rate limit, a timeout, an HTTP error, or the remote service being unreachable — the app automatically catches it and reruns the request on the locally executed model (`Qwen/Qwen3-0.6B`) instead, with no action required from the user. Every response is labeled with which model actually produced it (🌐 Remote API vs. 🖥️ Local Model), and a fallback response is additionally flagged with a "Remote API unavailable — automatically switched" notice.

The "Use Local Model" checkbox remains as a manual override for forcing local-only inference. A separate "Simulate Remote Outage (Demo Failover)" checkbox is provided purely to demonstrate the failover deterministically — enabling it raises a simulated error before the API call, exercising the exact same fallback path a real outage would trigger, without needing to wait for an actual rate limit or network failure.

**To see it work:** log in with Hugging Face, leave both checkboxes off, and send a message — the response is labeled 🌐 Remote API. Check "Simulate Remote Outage," send another message, and the response is labeled 🖥️ Local Model with the failover notice, while "Use Local Model" is still unchecked.

**Advantages:** the app stays available even when the remote model is rate-limited, slow, or down, and the user never has to notice or intervene. **Disadvantages:** the local model is smaller/weaker than the remote one, so a silent fallback can quietly degrade response quality; on a Space without a GPU, local inference is also considerably slower, so failover trades availability for latency and quality.
