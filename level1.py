# phi3_empathy_level1_fixed_en.py
# Phi-3-mini-4k-instruct + Level-1 Artificial Empathy

import re
import builtins
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"


# -----------------------------
# 1) Level-1 empathy policy
# -----------------------------
SYSTEM_EMPATHY_POLICY = """You are a helpful assistant.
Special mode: EMPATHY_LEVEL_1.

Rules:
- Reflect the user's emotion and validate it in 1-2 sentences (e.g., "That sounds really tough.").
- Then ask exactly ONE open-ended question (e.g., "Do you want to tell me more about what happened?").
- If the user clearly asks for advice/solutions, add ONLY ONE small, safe suggestion (breathing, a tiny next step, writing it down, taking a short break).
- Keep it under 6-8 sentences. Be concise and calm.
- Do not roleplay as a therapist. Do not diagnose.
- Do not encourage dangerous or harmful actions.
"""


# -----------------------------
# 2) Lightweight detection
# -----------------------------
EMOTION_PATTERNS = {
    "sad": [
        r"\bsad\b", r"\bdepressed\b", r"\bdown\b", r"\bheartbroken\b", r"\blonely\b", r"\bcrying\b"
    ],
    "stressed": [
        r"\bstress(ed)?\b", r"\boverwhelmed\b", r"\bburned out\b", r"\bpressure\b", r"\bpanicking\b"
    ],
    "angry": [
        r"\bangry\b", r"\bfurious\b", r"\bmad\b", r"\birritated\b", r"\brage\b"
    ],
    "anxious": [
        r"\banxious\b", r"\bworried\b", r"\bscared\b", r"\bafraid\b", r"\bnervous\b"
    ],
    "happy": [
        r"\bhappy\b", r"\bexcited\b", r"\bthrilled\b", r"\bproud\b", r"\bgrateful\b"
    ],
    "tired": [
        r"\btired\b", r"\bexhausted\b", r"\bno energy\b", r"\bsleep deprived\b", r"\bworn out\b"
    ],
}

ADVICE_CUES = [
    r"\bwhat should i do\b",
    r"\bwhat can i do\b",
    r"\bany advice\b",
    r"\bhelp me\b",
    r"\bhow do i\b",
    r"\bsolution\b",
    r"\btips\b",
]

FALLBACK_OPEN_QUESTION = "What part of this feels the hardest right now?"


def detect_emotion(text: str) -> str:
    t = text.lower()
    for emotion, patterns in EMOTION_PATTERNS.items():
        if any(re.search(p, t) for p in patterns):
            return emotion
    return "neutral"


def user_wants_advice(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in ADVICE_CUES)


def build_empathy_hint(user_text: str) -> str:
    emotion = detect_emotion(user_text)
    wants_advice = user_wants_advice(user_text)
    return (
        f"EMPATHY_HINT: emotion={emotion}, user_wants_advice={wants_advice}. "
        f"Follow EMPATHY_LEVEL_1 rules."
    )


# -----------------------------
# 3) Post-check / shaping
# -----------------------------
def enforce_level1_constraints(reply: str) -> str:
    """
    Enforce:
      - <= 8 sentences
      - exactly 1 question
    Lightweight (no extra model call).
    """
    r = reply.strip()

    # Normalize whitespace
    r = re.sub(r"\s+", " ", r).strip()

    # Sentence split (simple)
    # Keep delimiters by splitting on punctuation boundaries.
    chunks = re.split(r"(?<=[.!?])\s+", r)
    chunks = [c.strip() for c in chunks if c.strip()]

    # Truncate to max 8 sentences
    if len(chunks) > 8:
        chunks = chunks[:8]

    # Ensure exactly one question sentence.
    q_indices = [i for i, c in enumerate(chunks) if "?" in c]

    if len(q_indices) == 0:
        # Add a question at the end (counts as one sentence)
        if len(chunks) < 8:
            chunks.append(FALLBACK_OPEN_QUESTION)
        else:
            # Replace last sentence with the question
            chunks[-1] = FALLBACK_OPEN_QUESTION
    elif len(q_indices) > 1:
        # Keep the first question sentence, remove question marks from others
        first_q = q_indices[0]
        for i in q_indices[1:]:
            chunks[i] = chunks[i].replace("?", ".")
        # If the first question sentence has multiple ?, keep only one
        chunks[first_q] = re.sub(r"\?+", "?", chunks[first_q])

    # Re-join
    out = " ".join(chunks).strip()

    # Final clamp on sentence count after edits
    chunks2 = re.split(r"(?<=[.!?])\s+", out)
    chunks2 = [c.strip() for c in chunks2 if c.strip()]
    if len(chunks2) > 8:
        out = " ".join(chunks2[:8]).strip()

    return out


# -----------------------------
# 4) Model loading (FIXED)
# -----------------------------
def load_phi3(model_id: str = MODEL_ID):
    """
    Preferred: trust_remote_code=False to avoid cache API mismatches.
    Fallback: trust_remote_code=True but disable KV-cache in generation.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "auto" if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Try built-in implementation first (most stable)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if device == "cuda" else None,
            dtype=dtype,                 # torch_dtype deprecated -> dtype
            trust_remote_code=False,     # IMPORTANT: avoid legacy remote code
        )
        model.eval()
        return model, tokenizer, False  # use_cache_allowed=True (we can use cache)
    except Exception as e:
        print(f"[WARN] Built-in load failed ({type(e).__name__}: {e}).")
        print("[WARN] Falling back to trust_remote_code=True with KV-cache DISABLED.")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if device == "cuda" else None,
            dtype=dtype,
            trust_remote_code=True,
        )
        model.eval()

        # Disable cache globally to avoid DynamicCache/seen_tokens issues
        try:
            model.config.use_cache = False
        except Exception:
            pass

        return model, tokenizer, True  # cache_disabled=True


# -----------------------------
# 5) Chat formatting + generation
# -----------------------------
def apply_chat_template_safe(tokenizer, messages):
    """
    Use tokenizer.apply_chat_template when available.
    Fallback to a simple format if not.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Fallback formatting (rare)
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"{role.upper()}: {content}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)


@torch.inference_mode()
def generate_reply(model, tokenizer, history, user_text: str, cache_disabled: bool, max_new_tokens: int = 220) -> str:
    empathy_hint = build_empathy_hint(user_text)

    messages = [
        {"role": "system", "content": SYSTEM_EMPATHY_POLICY},
        {"role": "system", "content": empathy_hint},
        *history,
        {"role": "user", "content": user_text},
    ]

    prompt = apply_chat_template_safe(tokenizer, messages)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.08,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Critical fix when using remote code fallback: disable KV cache
    if cache_disabled:
        gen_kwargs["use_cache"] = False

    output_ids = model.generate(**inputs, **gen_kwargs)

    # Decode ONLY the newly generated tokens
    new_tokens = output_ids[0, input_len:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Mild cleanup
    reply = re.sub(r"^\s*(assistant:)\s*", "", reply, flags=re.I).strip()

    # Enforce Level-1 constraints (one question, <=8 sentences)
    reply = enforce_level1_constraints(reply)

    return reply


# -----------------------------
# 6) CLI loop
# -----------------------------
def main():
    model, tokenizer, cache_disabled = load_phi3()

    history = []  # list[{"role": "user"/"assistant", "content": "..."}]
    print("Phi-3 + EMPATHY_LEVEL_1 (type 'exit' to quit)\n")

    while True:
        # builtins.input avoids the "input is a module" problem
        user_text = builtins.input("You: ").strip()
        if user_text.lower() in ["exit", "quit"]:
            print("Assistant: Bye!")
            break

        reply = generate_reply(model, tokenizer, history, user_text, cache_disabled=cache_disabled)
        print("\nAssistant:", reply, "\n")

        # Keep a small window of history (avoid context bloat)
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})
        history = history[-8:]  # last 4 turns


if __name__ == "__main__":
    main()

