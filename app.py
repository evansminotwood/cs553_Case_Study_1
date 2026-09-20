import gradio as gr
import spaces
from huggingface_hub import InferenceClient
from transformers import pipeline

LOCAL_MODEL = "Qwen/Qwen3-0.6B"
REMOTE_MODEL = "openai/gpt-oss-20b"

pipe = pipeline(
    "text-generation",
    model=LOCAL_MODEL,
    dtype="auto",
    device="cuda",
)

fancy_css = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&display=swap');

:root, .dark {
    --body-background-fill: radial-gradient(ellipse at top, #145a32 0%, #0b3d2e 55%, #062017 100%);
    --background-fill-primary: #0f4a33;
    --background-fill-secondary: #0b3d2e;
    --border-color-primary: #d4af37;
    --border-color-accent: #d4af37;
    --button-primary-background-fill: #d4af37;
    --button-primary-background-fill-hover: #e8c766;
    --button-primary-text-color: #1a1006;
    --color-accent: #d4af37;
    --body-text-color: #f5f0e1;
    --link-text-color: #f4d160;
}

.gradio-container {
    width: 96% !important;
    max-width: none !important;
}
#app-title {
    text-align: center;
    margin-bottom: 4px;
    font-family: "Playfair Display", Georgia, serif;
    font-size: 2.4em;
    color: #d4af37;
    text-shadow: 0 2px 6px rgba(0, 0, 0, 0.5);
    letter-spacing: 1px;
}
#app-subtitle {
    text-align: center;
    color: #e8dcb5;
    font-style: italic;
    margin-bottom: 24px;
}
#chat-container {
    width: 100%;
    border: 2px solid #d4af37;
    border-radius: 16px;
    padding: 16px;
    background: repeating-linear-gradient(135deg, #0b3d2e, #0b3d2e 24px, #0e4534 24px, #0e4534 48px);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4), inset 0 0 40px rgba(0, 0, 0, 0.25);
}
#model-note {
    font-size: 0.9em;
    color: #e8dcb5;
    margin-top: 8px;
}
@media (max-width: 768px) {
    .gradio-container {
        width: 98% !important;
    }
    #chat-container {
        padding: 8px;
    }
    #app-title {
        font-size: 1.8em;
    }
}
"""


@spaces.GPU
def local_generate(
    messages,
    max_tokens,
    temperature,
    top_p,
):
    outputs = pipe(
        messages,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )

    return outputs[0]["generated_text"][-1]["content"]


def respond(
    message,
    history: list[dict[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
    use_local_model,
    simulate_remote_outage,
    hf_token: gr.OAuthToken,
):
    messages = [{"role": "system", "content": system_message}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    if use_local_model:
        print("[MODE] local (manual)")

        response = local_generate(
            messages,
            max_tokens,
            temperature,
            top_p,
        )

        yield f"🖥️ **[Local Model — {LOCAL_MODEL}]**\n\n{response}"
        return

    if hf_token is None or not getattr(hf_token, "token", None):
        yield "⚠️ Please log in with your Hugging Face account first."
        return

    print("[MODE] api (with automatic failover)")

    try:
        if simulate_remote_outage:
            raise RuntimeError("Simulated remote outage (demo toggle enabled)")

        client = InferenceClient(
            token=hf_token.token,
            model=REMOTE_MODEL,
            timeout=15,
        )

        response = ""

        for chunk in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            stream=True,
            temperature=temperature,
            top_p=top_p,
        ):
            choices = chunk.choices
            token = ""

            if len(choices) and choices[0].delta.content:
                token = choices[0].delta.content

            response += token
            yield f"🌐 **[Remote API — {REMOTE_MODEL}]**\n\n{response}"

        return
    except Exception as exc:
        print(f"[FAILOVER] Remote API unavailable ({exc!r}); falling back to local model")

    response = local_generate(messages, max_tokens, temperature, top_p)
    yield (
        "⚠️ **Remote API unavailable — automatically switched to the local model.**\n\n"
        f"🖥️ **[Local Model — {LOCAL_MODEL}, fallback]**\n\n{response}"
    )


def build_demo():
    chatbot = gr.ChatInterface(
        fn=respond,
        additional_inputs=[
            gr.Textbox(
                value=(
                    "You are a friendly, patient Blackjack tutor. For every hand, state the "
                    "mathematically optimal basic-strategy play (hit, stand, double down, split, "
                    "or surrender) and explain *why* it's correct in terms of the dealer's "
                    "up-card and the house edge. Adapt your explanations to the player's stated "
                    "skill level, and encourage responsible bankroll management."
                ),
                label="System message",
            ),
            gr.Slider(
                minimum=1,
                maximum=2048,
                value=512,
                step=1,
                label="Max new tokens",
            ),
            gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="Temperature",
            ),
            gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.95,
                step=0.05,
                label="Top-p (nucleus sampling)",
            ),
            gr.Checkbox(
                label="Use Local Model",
                value=False,
            ),
            gr.Checkbox(
                label="Simulate Remote Outage (Demo Failover)",
                value=False,
            ),
        ],
    )

    with gr.Blocks(css=fancy_css) as demo:
        with gr.Sidebar():
            gr.LoginButton()

        gr.Markdown(
            "# ♠️ Blackjack Tutor ♥️",
            elem_id="app-title",
        )

        gr.Markdown(
            "Master basic strategy, one hand at a time.",
            elem_id="app-subtitle",
        )

        with gr.Column(elem_id="chat-container"):
            chatbot.render()

            gr.Markdown(
                "By default, requests go to the remote API and automatically fail over to the "
                "local model if it's unavailable. Use **Additional inputs** to force the local "
                "model, or simulate a remote outage to see the failover in action.",
                elem_id="model-note",
            )

    return demo


if __name__ == "__main__":
    build_demo().launch()
