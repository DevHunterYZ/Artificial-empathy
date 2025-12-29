# phi3_empathy_level2_fixed_en_v4.py
# Phi-3-mini-4k-instruct + EMPATHY_LEVEL_2 (2-pass: classify -> respond) + Greeting router + Safe crisis gating v4

import re
import json
import builtins
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"


# =========================
# 1) SYSTEM POLICIES
# =========================
SYSTEM_EMPATHY_L2_POLICY = """You are a helpful assistant.
Special mode: EMPATHY_LEVEL_2.

Hard constraints:
- Do NOT invent context. If the user didn't mention hardship, do not imply "rough patches", "you need support", etc.
- Do not diagnose. Do not roleplay as a therapist.

Rules:
1) Start with 1–2 sentences that reflect the user's emotion + situation and validate it (no diagnosis).
2) Ask exactly ONE open-ended question (prefer "What/How" questions, not yes/no).
3) Only if the user clearly wants advice/solutions, add EXACTLY ONE small, safe, practical suggestion.
4) If the user is venting and NOT asking for advice, do not give advice; focus on listening and clarifying.
5) Keep it under 9 sentences.
"""

SYSTEM_SMALLTALK_POLICY = """You are a helpful assistant.
Special mode: SMALLTALK_FRIENDLY.

Hard constraint:
- Do NOT invent problems or hardship. Keep it light and friendly.

Rules:
- Respond naturally to greetings.
- If the user asks "how are you", answer briefly and positively.
- Ask exactly ONE open-ended question back.
- Keep it short (<= 4 sentences).
"""

SYSTEM_SAFETY_POLICY = """You are a helpful assistant.
Special mode: SAFETY_SUPPORT.

Hard constraints:
- Do NOT provide instructions for self-harm or violence.
- Do NOT invent crisis. Only respond in safety mode when the user indicates danger.

Rules:
- Be calm and supportive.
- Encourage immediate help from local emergency services or trusted people if they are in danger.
- Ask exactly ONE open-ended question focused on immediate safety.
- Keep it short (<= 7 sentences).
"""


# =========================
# 2) GREETING + POSITIVE/NEGATIVE EVIDENCE
# =========================
GREETINGS = {
    "hi", "hello", "hey", "yo",
    "good morning", "good afternoon", "good evening",
    "hi there", "hello there", "hey there",
}

POSITIVE_PATTERNS = [
    r"\bhappy\b", r"\bso happy\b", r"\bexcited\b", r"\bthrilled\b", r"\bgrateful\b",
    r"\bgreat\b", r"\bamazing\b", r"\bfantastic\b", r"\bgood news\b", r"\bawesome\b",
    r":\)", r":d", r"😊", r"😁", r"😄", r"😍", r"🥳"
]

NEGATIVE_EVIDENCE_PATTERNS = [
    r"\bsuicide\b", r"\bkill myself\b", r"\bself[- ]harm\b", r"\bhurt myself\b",
    r"\bi want to die\b", r"\bno reason to live\b", r"\bcan't go on\b",
    r"\bpanic\b", r"\boverwhelmed\b", r"\bdepressed\b", r"\bhopeless\b",
    r"\bunsafe\b", r"\bin danger\b", r"\bthreat\b"
]

def looks_like_greeting(text: str) -> bool:
    t = re.sub(r"[^a-z\s]", "", text.lower()).strip()
    if t in GREETINGS:
        return True
    if t.startswith(("hi ", "hello ", "hey ")):
        if len(t) <= 60 and any(x in t for x in ["how are you", "how r u", "what's up", "hows it going", "how are u"]):
            return True
    return False

def is_clearly_positive(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in POSITIVE_PATTERNS)

def has_negative_evidence(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in NEGATIVE_EVIDENCE_PATTERNS)


# =========================
# 3) SAFETY / CRISIS DETECT (STRICT)
# =========================
SELF_HARM_STRONG = [
    r"\bi want to die\b",
    r"\bi'm suicidal\b",
    r"\bi am suicidal\b",
    r"\bkill myself\b",
    r"\bsuicide\b",
    r"\bself[- ]harm\b",
    r"\bhurt myself\b",
    r"\bend it all\b",
    r"\bno reason to live\b",
    r"\bcan't go on\b",
]

VIOLENCE_STRONG = [
    r"\bi will kill\b",
    r"\bi'm going to kill\b",
    r"\bi am going to kill\b",
    r"\bi will hurt\b",
    r"\bi'm going to hurt\b",
    r"\bi am going to hurt\b",
]

def regex_crisis(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in (SELF_HARM_STRONG + VIOLENCE_STRONG))


# =========================
# 4) LIGHTWEIGHT CONTEXT SUMMARY
# =========================
def make_context_summary(history: List[Dict[str, str]], max_chars: int = 240) -> str:
    user_turns = [m["content"] for m in history if m.get("role") == "user"]
    recent = user_turns[-2:] if len(user_turns) >= 2 else user_turns
    summary = " | ".join(re.sub(r"\s+", " ", x).strip() for x in recent if x.strip())
    return summary[:max_chars] if summary else "(no prior context)"


# =========================
# 5) 2-PASS CLASSIFIER
# =========================
CLASSIFIER_SYSTEM = """You are a strict classifier.
Output ONLY valid minified JSON with these keys:
emotion (one of: sad, stressed, angry, anxious, happy, tired, neutral),
intent (one of: greeting, venting, advice, decision, reassurance, information),
wants_advice (true/false),
urgency (one of: low, medium, high),
topic (short string, 1-4 words).

Important constraints:
- Do NOT invent crisis.
- Set urgency="high" ONLY if the user explicitly mentions self-harm, suicide, violence, or immediate danger.
- If the message is clearly positive (e.g., "I'm so happy :)"), do NOT set urgency high and do NOT use safety-related topic.

No extra text. No markdown. JSON only.
"""

CLASSIFIER_USER_TEMPLATE = """Classify the user's message.

User message:
{user_text}
"""

@dataclass
class ClassifyResult:
    emotion: str = "neutral"
    intent: str = "venting"
    wants_advice: bool = False
    urgency: str = "low"
    topic: str = "general"

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ClassifyResult":
        def norm_str(x, default):
            return str(x).strip().lower() if x is not None else default

        emotion = norm_str(d.get("emotion"), "neutral")
        intent = norm_str(d.get("intent"), "venting")
        urgency = norm_str(d.get("urgency"), "low")
        topic = str(d.get("topic") or "general").strip().lower()

        wants = d.get("wants_advice")
        if isinstance(wants, bool):
            wants_advice = wants
        else:
            wants_advice = str(wants).strip().lower() in ("true", "1", "yes")

        if emotion not in {"sad","stressed","angry","anxious","happy","tired","neutral"}:
            emotion = "neutral"
        if intent not in {"greeting","venting","advice","decision","reassurance","information"}:
            intent = "venting"
        if urgency not in {"low","medium","high"}:
            urgency = "low"
        if not topic:
            topic = "general"

        return ClassifyResult(emotion, intent, wants_advice, urgency, topic)

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    t = text.strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", t, flags=re.S)
    if not m:
        return None

    candidate = m.group(0).strip()
    candidate = re.sub(r",\s*}", "}", candidate)
    candidate = re.sub(r",\s*]", "]", candidate)

    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# =========================
# 6) POST-CHECK v2
# =========================
OPEN_Q_START = ("what", "how", "why", "when", "where", "which", "tell me", "walk me through", "in what way")
FALLBACK_OPEN_QUESTION_L2 = "What part of this feels most important to talk about right now?"

VALIDATION_MARKERS = [
    "that makes sense", "understandable", "it makes sense", "sounds really", "sounds tough",
    "i can see why", "i hear you", "that’s a lot", "that is a lot", "would be hard"
]

def sentence_split(text: str) -> List[str]:
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    return [c.strip() for c in chunks if c.strip()]

def has_validation(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in VALIDATION_MARKERS)

def question_sentences(sentences: List[str]) -> List[Tuple[int, str]]:
    return [(i, s) for i, s in enumerate(sentences) if "?" in s]

def is_open_question(q: str) -> bool:
    q2 = q.strip().lower()
    q2 = re.sub(r"^[\"'“”‘’]+", "", q2).strip()
    return q2.startswith(OPEN_Q_START)

def enforce_level2_constraints(reply: str, wants_advice: bool, crisis_mode: bool) -> str:
    r = re.sub(r"\s+", " ", reply).strip()
    sents = sentence_split(r)

    max_sents = 7 if crisis_mode else 9
    if len(sents) > max_sents:
        sents = sents[:max_sents]

    joined = " ".join(sents)
    if not crisis_mode and not has_validation(joined):
        prefix = "That makes sense, and I hear you."
        sents = [prefix] + sents
        if len(sents) > max_sents:
            sents = sents[:max_sents]

    qs = question_sentences(sents)
    if len(qs) == 0:
        if len(sents) < max_sents:
            sents.append(FALLBACK_OPEN_QUESTION_L2)
        else:
            sents[-1] = FALLBACK_OPEN_QUESTION_L2
    else:
        first_idx, _ = qs[0]
        for idx, _ in qs[1:]:
            sents[idx] = sents[idx].replace("?", ".")
        if not is_open_question(sents[first_idx]):
            sents[first_idx] = FALLBACK_OPEN_QUESTION_L2
        for i in range(len(sents)):
            if i != first_idx:
                sents[i] = sents[i].replace("?", ".")

    def is_suggestion_like(s: str) -> bool:
        sl = s.lower()
        return any(k in sl for k in [
            "you could", "try to", "i suggest", "one thing you can do", "here's what to do",
            "a good idea is", "step", "plan:"
        ])

    if crisis_mode:
        sents = [s for s in sents if not is_suggestion_like(s)]
        if not sents:
            sents = ["I’m really sorry you’re going through this. You deserve support right now."]
        if "?" not in " ".join(sents):
            q = "What’s the safest thing you can do right now—are you with someone you trust?"
            if len(sents) < max_sents:
                sents.append(q)
            else:
                sents[-1] = q
        safety_line = "If you’re in immediate danger, please contact local emergency services or a trusted person right now."
        if all("emergency" not in s.lower() and "immediate danger" not in s.lower() for s in sents):
            if len(sents) < max_sents:
                sents.insert(min(2, len(sents)), safety_line)
            else:
                sents[-2] = safety_line
        return " ".join(sents).strip()

    if not wants_advice:
        sents2 = [s for s in sents if not is_suggestion_like(s)]
        if len(sents2) >= 2:
            sents = sents2[:max_sents]
    else:
        kept = []
        suggestion_kept = False
        for s in sents:
            if is_suggestion_like(s):
                if not suggestion_kept:
                    kept.append("One small thing you could try is to take a slow breath and pick just one tiny next step.")
                    suggestion_kept = True
            else:
                kept.append(s)
        sents = kept[:max_sents]

    return " ".join(sents).strip()


# =========================
# 7) MODEL LOADING (STABLE)
# =========================
def load_phi3(model_id: str = MODEL_ID):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = "auto" if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if device == "cuda" else None,
            dtype=dtype,
            trust_remote_code=False,
        )
        model.eval()
        return model, tokenizer, False
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
        try:
            model.config.use_cache = False
        except Exception:
            pass
        return model, tokenizer, True


# =========================
# 8) GENERATION (NEW TOKENS) - no warning
# =========================
def apply_chat_template_safe(tokenizer, messages):
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    parts = []
    for m in messages:
        parts.append(f"{m.get('role','user').upper()}: {m.get('content','')}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)

@torch.inference_mode()
def generate_new_tokens_text(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    cache_disabled: bool,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.08
) -> str:
    prompt = apply_chat_template_safe(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)
    if cache_disabled:
        gen_kwargs["use_cache"] = False

    out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0, input_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    text = re.sub(r"^\s*(assistant:)\s*", "", text, flags=re.I).strip()
    return text


# =========================
# 9) LEVEL 2 PIPELINE
# =========================
def classify_message(model, tokenizer, user_text: str, cache_disabled: bool) -> "ClassifyResult":
    # Heuristic greeting shortcut
    if looks_like_greeting(user_text) and len(user_text.strip()) <= 80:
        return ClassifyResult(emotion="neutral", intent="greeting", wants_advice=False, urgency="low", topic="greeting")

    # Heuristic positive shortcut (prevents weird "high urgency" outputs)
    if is_clearly_positive(user_text) and len(user_text.strip()) <= 120:
        return ClassifyResult(emotion="happy", intent="reassurance", wants_advice=False, urgency="low", topic="positive")

    messages = [
        {"role": "system", "content": CLASSIFIER_SYSTEM},
        {"role": "user", "content": CLASSIFIER_USER_TEMPLATE.format(user_text=user_text.strip())},
    ]

    raw = generate_new_tokens_text(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        cache_disabled=cache_disabled,
        max_new_tokens=120,
        do_sample=False,
        repetition_penalty=1.0
    )

    obj = extract_json_object(raw)
    if not obj:
        return ClassifyResult(emotion="neutral", intent="venting", wants_advice=False, urgency="low", topic="general")
    cls = ClassifyResult.from_dict(obj)

    # If user text is clearly positive, clamp classifier just in case
    if is_clearly_positive(user_text):
        cls.urgency = "low"
        if cls.intent != "greeting":
            cls.intent = "reassurance"
        cls.topic = "positive"
        cls.emotion = "happy"
        cls.wants_advice = False

    return cls

def should_enter_crisis_mode(user_text: str, cls: "ClassifyResult") -> bool:
    # Hard blocks
    if looks_like_greeting(user_text) and len(user_text.strip()) <= 120:
        return False
    if is_clearly_positive(user_text):
        return False

    # Strong regex always wins
    if regex_crisis(user_text):
        return True

    # Classifier-based crisis ONLY if there is negative evidence in the text too
    safety_topics = {"safety", "self-harm", "selfharm", "suicide", "violence", "danger"}
    if cls.urgency == "high" and any(t in cls.topic for t in safety_topics) and has_negative_evidence(user_text):
        return True

    return False

def build_responder_messages(
    history: List[Dict[str, str]],
    user_text: str,
    cls: "ClassifyResult",
    crisis_mode: bool
) -> List[Dict[str, str]]:
    context_summary = make_context_summary(history)

    if crisis_mode:
        system_policy = SYSTEM_SAFETY_POLICY
    elif cls.intent == "greeting" or cls.topic == "greeting":
        system_policy = SYSTEM_SMALLTALK_POLICY
    else:
        system_policy = SYSTEM_EMPATHY_L2_POLICY

    hint = (
        f"CONTEXT_SUMMARY: {context_summary}\n"
        f"CLASSIFY_JSON: {json.dumps(cls.__dict__, separators=(',', ':'))}\n"
        f"INSTRUCTION: Follow the active system policy. Do NOT invent context. "
        f"Ask exactly one open-ended question."
    )

    return [
        {"role": "system", "content": system_policy},
        {"role": "system", "content": hint},
        *history,
        {"role": "user", "content": user_text.strip()},
    ]

def level2_reply(model, tokenizer, history, user_text: str, cache_disabled: bool) -> str:
    cls = classify_message(model, tokenizer, user_text, cache_disabled)
    crisis_mode = should_enter_crisis_mode(user_text, cls)

    if crisis_mode:
        cls = ClassifyResult(emotion="neutral", intent="reassurance", wants_advice=False, urgency="high", topic="safety")

    messages = build_responder_messages(history, user_text, cls, crisis_mode)

    raw_reply = generate_new_tokens_text(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        cache_disabled=cache_disabled,
        max_new_tokens=200 if (cls.intent == "greeting") else (220 if cls.emotion == "happy" else (260 if not crisis_mode else 180)),
        do_sample=True,
        temperature=0.7 if not crisis_mode else 0.4,
        top_p=0.9,
        repetition_penalty=1.08
    )

    # Greeting: enforce simple one-question + <=4 sentences, no validation forcing
    if cls.intent == "greeting":
        sents = sentence_split(re.sub(r"\s+", " ", raw_reply).strip())[:4]
        qs = question_sentences(sents)
        if len(qs) == 0:
            sents.append("How are you doing today?")
        else:
            first_idx, _ = qs[0]
            for idx, _ in qs[1:]:
                sents[idx] = sents[idx].replace("?", ".")
            for i in range(len(sents)):
                if i != first_idx:
                    sents[i] = sents[i].replace("?", ".")
        return " ".join(sents).strip()

    wants_advice = bool(cls.wants_advice) and (cls.intent in {"advice", "decision"})
    return enforce_level2_constraints(raw_reply, wants_advice=wants_advice, crisis_mode=crisis_mode)


# =========================
# 10) CLI LOOP
# =========================
def main():
    model, tokenizer, cache_disabled = load_phi3()
    history: List[Dict[str, str]] = []

    print("Phi-3 + EMPATHY_LEVEL_2 (fixed v4) (type 'exit' to quit)\n")

    while True:
        user_text = builtins.input("You: ").strip()
        if user_text.lower() in ("exit", "quit"):
            print("Assistant: Bye!")
            break

        reply = level2_reply(model, tokenizer, history, user_text, cache_disabled)
        print("\nAssistant:", reply, "\n")

        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})
        history = history[-8:]  # last 4 turns


if __name__ == "__main__":
    main()
