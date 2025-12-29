# phi3_empathy_level3_fixed_en.py
# Phi-3-mini-4k-instruct + EMPATHY_LEVEL_3

import re
import json
import time
import builtins
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
MEMORY_PATH = "empathy_memory.json"


# =========================
# 0) SKILL SET (LEVEL-3)
# =========================
SKILLS = [
    "reflect_label",        # reflect situation + label emotion
    "validate_normalize",   # normalize: "that makes sense given X"
    "probe_needs",          # needs/values question
    "clarify_context",      # what happened / what led to this
    "decision_support",     # help choose between options (only if asked)
    "micro_reframe",        # gentle reframe without invalidation
    "celebrate_amplify",    # positive celebration + amplification
    "boundary_autonomy",    # ask preference: listen vs ideas
    "micro_action",         # ONE tiny safe suggestion (only if allowed)
]


# =========================
# 1) POLICIES
# =========================
SYSTEM_EMPATHY_L3_POLICY = """You are a helpful assistant.
Special mode: EMPATHY_LEVEL_3.

Hard constraints:
- Do NOT invent context. Never imply hardship or danger unless the user stated it.
- Do not diagnose. Do not roleplay as a therapist.
- Ask exactly ONE open-ended question (prefer What/How). No yes/no questions.
- Only give advice if allowed by the plan. If advice is not allowed, do NOT give suggestions.

Response structure:
1) 1–2 sentences: accurate reflection + emotion label (grounded in the user's message).
2) 1 sentence: validation/normalization ("Given X, it makes sense...") if appropriate.
3) Optional: ONE tiny, safe suggestion ONLY if allowed by the plan (and keep it as a single sentence).
4) Exactly ONE open-ended question aligned to the plan.

Keep it concise: <= 10 sentences.
"""

SYSTEM_SMALLTALK_POLICY = """You are a helpful assistant.
Special mode: SMALLTALK_FRIENDLY.

Hard constraint:
- Do NOT invent problems or hardship.

Rules:
- Respond naturally to greetings.
- If the user asks "how are you", answer briefly and positively.
- Ask exactly ONE open-ended question back.
- Keep it <= 4 sentences.
"""

SYSTEM_SAFETY_POLICY = """You are a helpful assistant.
Special mode: SAFETY_SUPPORT.

Hard constraints:
- Do NOT provide instructions for self-harm or violence.
- Do NOT invent crisis.

Rules:
- Be calm and supportive.
- Encourage immediate help from local emergency services or trusted people if they are in danger.
- Ask exactly ONE open-ended question focused on immediate safety.
- Keep it <= 7 sentences.
"""

# Rewrite policy to remove invented context / overreach
SYSTEM_REWRITE_POLICY = """You are a careful editor.
Goal: rewrite the assistant draft so it is strictly grounded in the USER MESSAGE and CONTEXT SUMMARY only.

Rules:
- Remove any implied hardship, danger, or background facts not stated by the user or context.
- Keep the same tone and keep it helpful.
- Keep exactly ONE open-ended question.
- If advice is not allowed, remove suggestions.
- Output only the rewritten text.
"""


# =========================
# 2) GREETING / POSITIVE / NEGATIVE EVIDENCE
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
        if len(t) <= 80 and any(x in t for x in ["how are you", "how r u", "what's up", "hows it going", "how are u"]):
            return True
    return False

def is_clearly_positive(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in POSITIVE_PATTERNS)

def has_negative_evidence(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in NEGATIVE_EVIDENCE_PATTERNS)


# =========================
# 3) CRISIS DETECT (STRICT)
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
# 4) MEMORY
# =========================
@dataclass
class EmpathyMemory:
    # preference: "auto" | "listen" | "ideas"
    advice_preference: str = "auto"
    # lock listen/ideas for N turns if explicitly requested
    advice_lock_mode: str = "none"   # "none" | "listen" | "ideas"
    advice_lock_turns: int = 0

    # counters
    topic_counts: Dict[str, int] = None
    helped_actions: Dict[str, int] = None

    def __post_init__(self):
        if self.topic_counts is None:
            self.topic_counts = {}
        if self.helped_actions is None:
            self.helped_actions = {}

def load_memory(path: str = MEMORY_PATH) -> EmpathyMemory:
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        mem = EmpathyMemory(
            advice_preference=d.get("advice_preference", "auto"),
            advice_lock_mode=d.get("advice_lock_mode", "none"),
            advice_lock_turns=int(d.get("advice_lock_turns", 0)),
            topic_counts=d.get("topic_counts") or {},
            helped_actions=d.get("helped_actions") or {},
        )
        return mem
    except Exception:
        return EmpathyMemory()

def save_memory(mem: EmpathyMemory, path: str = MEMORY_PATH):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(mem), f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def update_memory_from_user(mem: EmpathyMemory, user_text: str):
    t = user_text.lower()

    # Explicit preference signals
    if re.search(r"\bjust listen\b|\bdon't give advice\b|\bno advice\b|\bjust hear me out\b", t):
        mem.advice_preference = "listen"
        mem.advice_lock_mode = "listen"
        mem.advice_lock_turns = max(mem.advice_lock_turns, 3)  # lock for next 3 turns
    elif re.search(r"\bgive me advice\b|\bgive me ideas\b|\bwhat should i do\b|\btell me what to do\b", t):
        mem.advice_preference = "ideas"
        mem.advice_lock_mode = "ideas"
        mem.advice_lock_turns = max(mem.advice_lock_turns, 3)

def decay_locks(mem: EmpathyMemory):
    if mem.advice_lock_turns > 0:
        mem.advice_lock_turns -= 1
        if mem.advice_lock_turns <= 0:
            mem.advice_lock_turns = 0
            mem.advice_lock_mode = "none"


# =========================
# 5) CONTEXT SUMMARY
# =========================
def make_context_summary(history: List[Dict[str, str]], max_chars: int = 260) -> str:
    user_turns = [m["content"] for m in history if m.get("role") == "user"]
    recent = user_turns[-2:] if len(user_turns) >= 2 else user_turns
    summary = " | ".join(re.sub(r"\s+", " ", x).strip() for x in recent if x.strip())
    return summary[:max_chars] if summary else "(no prior context)"


# =========================
# 6) CLASSIFY + PLAN DATA STRUCTS
# =========================
CLASSIFIER_SYSTEM = """You are a strict classifier.
Output ONLY valid minified JSON with these keys:
emotion (one of: sad, stressed, angry, anxious, happy, tired, neutral),
intent (one of: greeting, positive, venting, advice, decision, reassurance, information),
wants_advice (true/false),
urgency (one of: low, medium, high),
topic (short string, 1-4 words).

Important constraints:
- Do NOT invent crisis.
- Set urgency="high" ONLY if the user explicitly mentions self-harm, suicide, violence, or immediate danger.
- If the message is clearly positive, set intent="positive", emotion="happy", urgency="low".

No extra text. No markdown. JSON only.
"""

CLASSIFIER_USER_TEMPLATE = """Classify the user's message.

User message:
{user_text}
"""

PLANNER_SYSTEM = """You are an empathy response planner for EMPATHY_LEVEL_3.
Return ONLY valid minified JSON with keys:
selected_skills (array of 2 strings from the allowed list),
advice_allowed (true/false),
question_focus (short string),
tone (one of: light, calm, warm, celebratory),
topic (short string, 1-4 words).

Allowed skills:
reflect_label, validate_normalize, probe_needs, clarify_context, decision_support,
micro_reframe, celebrate_amplify, boundary_autonomy, micro_action

Rules:
- Always include reflect_label.
- If intent=positive -> include celebrate_amplify and tone=celebratory; advice_allowed=false.
- If wants_advice=false and memory_preference=listen -> advice_allowed=false; include boundary_autonomy or clarify_context.
- If wants_advice=true or memory_preference=ideas -> advice_allowed=true; include micro_action (and decision_support if intent=decision).
- Keep it conservative: avoid advice unless clearly allowed.

Output JSON only. No extra text.
"""

PLANNER_USER_TEMPLATE = """Plan a response.

CONTEXT_SUMMARY:
{context_summary}

CLASSIFY_JSON:
{classify_json}

MEMORY_PREF:
{memory_pref}

LOCK_MODE:
{lock_mode} (turns_left={lock_turns})

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
        if intent not in {"greeting","positive","venting","advice","decision","reassurance","information"}:
            intent = "venting"
        if urgency not in {"low","medium","high"}:
            urgency = "low"
        if not topic:
            topic = "general"

        return ClassifyResult(emotion, intent, wants_advice, urgency, topic)

@dataclass
class PlanResult:
    selected_skills: List[str]
    advice_allowed: bool
    question_focus: str
    tone: str
    topic: str

    @staticmethod
    def fallback_from(cls: ClassifyResult, mem: EmpathyMemory) -> "PlanResult":
        # Conservative default planner if parsing fails
        skills = ["reflect_label", "clarify_context"]
        tone = "warm"
        advice_allowed = False

        if cls.intent == "positive":
            skills = ["reflect_label", "celebrate_amplify"]
            tone = "celebratory"
            advice_allowed = False
        else:
            # honor lock
            if mem.advice_lock_mode == "ideas":
                advice_allowed = True
            elif mem.advice_lock_mode == "listen":
                advice_allowed = False
            else:
                advice_allowed = bool(cls.wants_advice) or (mem.advice_preference == "ideas")

            if advice_allowed:
                if cls.intent == "decision":
                    skills = ["reflect_label", "decision_support"]
                else:
                    skills = ["reflect_label", "micro_action"]
                tone = "calm"

        return PlanResult(skills, advice_allowed, "the key detail", tone, cls.topic or "general")


# =========================
# 7) JSON EXTRACTION
# =========================
def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    t = text.strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
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
# 8) POST-CHECK (ONE OPEN QUESTION, LENGTH, ADVICE)
# =========================
OPEN_Q_START = ("what", "how", "why", "when", "where", "which", "tell me", "walk me through", "in what way")
FALLBACK_OPEN_QUESTION = "What feels most important to focus on right now?"

def sentence_split(text: str) -> List[str]:
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    return [c.strip() for c in chunks if c.strip()]

def question_sentences(sentences: List[str]) -> List[Tuple[int, str]]:
    return [(i, s) for i, s in enumerate(sentences) if "?" in s]

def is_open_question(q: str) -> bool:
    q2 = q.strip().lower()
    q2 = re.sub(r"^[\"'“”‘’]+", "", q2).strip()
    return q2.startswith(OPEN_Q_START)

def enforce_one_open_question(text: str, max_sents: int) -> str:
    sents = sentence_split(re.sub(r"\s+", " ", text).strip())
    if len(sents) > max_sents:
        sents = sents[:max_sents]

    qs = question_sentences(sents)
    if len(qs) == 0:
        if len(sents) < max_sents:
            sents.append(FALLBACK_OPEN_QUESTION)
        else:
            sents[-1] = FALLBACK_OPEN_QUESTION
    else:
        first_idx, _ = qs[0]
        # remove other questions
        for idx, _ in qs[1:]:
            sents[idx] = sents[idx].replace("?", ".")
        # ensure open
        if not is_open_question(sents[first_idx]):
            sents[first_idx] = FALLBACK_OPEN_QUESTION
        # ensure only one ?
        for i in range(len(sents)):
            if i != first_idx:
                sents[i] = sents[i].replace("?", ".")

    return " ".join(sents).strip()

def strip_unsolicited_advice(text: str) -> str:
    # very lightweight heuristic: remove common suggestion sentences
    sents = sentence_split(text)
    def looks_like_suggestion(s: str) -> bool:
        sl = s.lower()
        return any(k in sl for k in [
            "you should", "you could", "try to", "i suggest", "one thing you can do",
            "here's what to do", "a good idea is", "step ", "plan:"
        ])
    kept = [s for s in sents if not looks_like_suggestion(s)]
    return " ".join(kept).strip() if kept else text.strip()


# =========================
# 9) ANTI-HALLUCINATED EMPATHY GUARD
# =========================
HARDSHIP_CUES = [
    "i'm sorry you're going through this",
    "you deserve support right now",
    "rough patch",
    "in danger",
    "it sounds like you're struggling",
    "that must be devastating",
]

def likely_invented_hardship(reply: str, user_text: str, context_summary: str, crisis_mode: bool) -> bool:
    if crisis_mode:
        return False  # safety mode is allowed to be supportive
    # If user is clearly positive, any hardship cues are suspicious
    if is_clearly_positive(user_text):
        rl = reply.lower()
        return any(cue in rl for cue in HARDSHIP_CUES)
    # If user has no negative evidence and reply has strong hardship cues, suspicious
    if not has_negative_evidence(user_text) and not has_negative_evidence(context_summary):
        rl = reply.lower()
        return any(cue in rl for cue in HARDSHIP_CUES)
    return False


# =========================
# 10) MODEL LOADING (STABLE)
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
# 11) GENERATION (NEW TOKENS) - no warning
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
    # Only pass temperature/top_p when sampling
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
# 12) PIPELINE: CLASSIFY -> PLAN -> RESPOND -> (OPTIONAL REWRITE)
# =========================
def classify_message(model, tokenizer, user_text: str, cache_disabled: bool) -> ClassifyResult:
    # Strong heuristics first
    if looks_like_greeting(user_text) and len(user_text.strip()) <= 120:
        return ClassifyResult(emotion="neutral", intent="greeting", wants_advice=False, urgency="low", topic="greeting")
    if is_clearly_positive(user_text) and len(user_text.strip()) <= 160:
        return ClassifyResult(emotion="happy", intent="positive", wants_advice=False, urgency="low", topic="positive")

    messages = [
        {"role": "system", "content": CLASSIFIER_SYSTEM},
        {"role": "user", "content": CLASSIFIER_USER_TEMPLATE.format(user_text=user_text.strip())},
    ]
    raw = generate_new_tokens_text(
        model, tokenizer, messages,
        cache_disabled=cache_disabled,
        max_new_tokens=140,
        do_sample=False,
        repetition_penalty=1.0
    )
    obj = extract_json_object(raw)
    if not obj:
        return ClassifyResult()
    cls = ClassifyResult.from_dict(obj)

    # Clamp positives just in case
    if is_clearly_positive(user_text):
        cls.emotion = "happy"
        cls.intent = "positive"
        cls.urgency = "low"
        cls.topic = "positive"
        cls.wants_advice = False

    return cls

def should_enter_crisis_mode(user_text: str, cls: ClassifyResult, context_summary: str) -> bool:
    # Hard blocks
    if looks_like_greeting(user_text):
        return False
    if is_clearly_positive(user_text):
        return False

    # Strong regex wins
    if regex_crisis(user_text):
        return True

    # Conservative additional gate: only if classifier is high AND negative evidence exists in user/context
    safety_topics = {"safety", "self-harm", "selfharm", "suicide", "violence", "danger"}
    if cls.urgency == "high" and any(t in cls.topic for t in safety_topics):
        if has_negative_evidence(user_text) or has_negative_evidence(context_summary):
            return True

    return False

def plan_response(model, tokenizer, user_text: str, cls: ClassifyResult, mem: EmpathyMemory,
                  history: List[Dict[str, str]], cache_disabled: bool) -> PlanResult:
    context_summary = make_context_summary(history)

    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": PLANNER_USER_TEMPLATE.format(
            context_summary=context_summary,
            classify_json=json.dumps(asdict(cls), separators=(",", ":")),
            memory_pref=mem.advice_preference,
            lock_mode=mem.advice_lock_mode,
            lock_turns=mem.advice_lock_turns,
            user_text=user_text.strip(),
        )},
    ]

    raw = generate_new_tokens_text(
        model, tokenizer, messages,
        cache_disabled=cache_disabled,
        max_new_tokens=180,
        do_sample=False,
        repetition_penalty=1.0
    )

    obj = extract_json_object(raw)
    if not obj:
        return PlanResult.fallback_from(cls, mem)

    # Validate plan
    skills = obj.get("selected_skills") or []
    skills = [str(s).strip().lower() for s in skills if str(s).strip()]
    skills = [s for s in skills if s in SKILLS]

    if "reflect_label" not in skills:
        skills = ["reflect_label"] + skills
    skills = skills[:2] if len(skills) >= 2 else (skills + ["validate_normalize"])[:2]

    advice_allowed = bool(obj.get("advice_allowed", False))
    tone = str(obj.get("tone") or "warm").strip().lower()
    if tone not in {"light", "calm", "warm", "celebratory"}:
        tone = "warm"
    question_focus = str(obj.get("question_focus") or "the key detail").strip()
    topic = str(obj.get("topic") or cls.topic or "general").strip().lower()

    # Enforce memory locks
    if mem.advice_lock_mode == "listen":
        advice_allowed = False
    elif mem.advice_lock_mode == "ideas":
        advice_allowed = True

    # Enforce intent rules
    if cls.intent == "positive":
        advice_allowed = False
        if "celebrate_amplify" not in skills:
            skills = ["reflect_label", "celebrate_amplify"]
        tone = "celebratory"

    return PlanResult(
        selected_skills=skills,
        advice_allowed=advice_allowed,
        question_focus=question_focus[:80],
        tone=tone,
        topic=topic[:40]
    )

def build_responder_messages(history: List[Dict[str, str]],
                             user_text: str,
                             cls: ClassifyResult,
                             plan: PlanResult,
                             mem: EmpathyMemory,
                             crisis_mode: bool) -> List[Dict[str, str]]:
    context_summary = make_context_summary(history)

    if crisis_mode:
        system_policy = SYSTEM_SAFETY_POLICY
    elif cls.intent == "greeting":
        system_policy = SYSTEM_SMALLTALK_POLICY
    else:
        system_policy = SYSTEM_EMPATHY_L3_POLICY

    hint = {
        "context_summary": context_summary,
        "classify": asdict(cls),
        "plan": asdict(plan),
        "memory": {
            "advice_preference": mem.advice_preference,
            "advice_lock_mode": mem.advice_lock_mode,
            "advice_lock_turns": mem.advice_lock_turns,
            "top_topics": sorted(mem.topic_counts.items(), key=lambda x: -x[1])[:5],
        }
    }

    return [
        {"role": "system", "content": system_policy},
        {"role": "system", "content": f"HINT_JSON: {json.dumps(hint, separators=(',', ':'))}"},
        *history,
        {"role": "user", "content": user_text.strip()},
    ]

def rewrite_if_needed(model, tokenizer, draft: str, user_text: str, context_summary: str,
                      advice_allowed: bool, cache_disabled: bool) -> str:
    # Always enforce one open question and advice gating via rewrite when needed
    messages = [
        {"role": "system", "content": SYSTEM_REWRITE_POLICY},
        {"role": "user", "content": (
            f"USER MESSAGE:\n{user_text}\n\n"
            f"CONTEXT SUMMARY:\n{context_summary}\n\n"
            f"ADVICE_ALLOWED: {advice_allowed}\n\n"
            f"ASSISTANT DRAFT:\n{draft}\n\n"
            f"Rewrite now."
        )}
    ]
    rewritten = generate_new_tokens_text(
        model, tokenizer, messages,
        cache_disabled=cache_disabled,
        max_new_tokens=240,
        do_sample=False,
        repetition_penalty=1.0
    )
    return rewritten.strip() if rewritten.strip() else draft

def level3_reply(model, tokenizer, history: List[Dict[str, str]], user_text: str,
                 mem: EmpathyMemory, cache_disabled: bool) -> str:
    update_memory_from_user(mem, user_text)

    context_summary = make_context_summary(history)
    cls = classify_message(model, tokenizer, user_text, cache_disabled)

    crisis_mode = should_enter_crisis_mode(user_text, cls, context_summary)
    if crisis_mode:
        # Safety override
        cls = ClassifyResult(emotion="neutral", intent="reassurance", wants_advice=False, urgency="high", topic="safety")
        plan = PlanResult(selected_skills=["reflect_label", "clarify_context"],
                          advice_allowed=False, question_focus="immediate safety",
                          tone="calm", topic="safety")
    else:
        plan = plan_response(model, tokenizer, user_text, cls, mem, history, cache_disabled)

    # Track topic in memory
    if plan.topic:
        mem.topic_counts[plan.topic] = int(mem.topic_counts.get(plan.topic, 0)) + 1

    # Build messages
    messages = build_responder_messages(history, user_text, cls, plan, mem, crisis_mode)

    # Generate
    if cls.intent == "greeting":
        raw = generate_new_tokens_text(
            model, tokenizer, messages,
            cache_disabled=cache_disabled,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05
        )
        # Simple enforcement: one open question, <=4 sentences
        out = enforce_one_open_question(raw, max_sents=4)
        return out

    raw = generate_new_tokens_text(
        model, tokenizer, messages,
        cache_disabled=cache_disabled,
        max_new_tokens=260 if not crisis_mode else 180,
        do_sample=True,
        temperature=0.7 if not crisis_mode else 0.4,
        top_p=0.9,
        repetition_penalty=1.08
    )

    # Enforce advice gating with heuristics first
    out = raw
    if not plan.advice_allowed and not crisis_mode:
        out = strip_unsolicited_advice(out)

    # Always enforce exactly one open question and length
    out = enforce_one_open_question(out, max_sents=(7 if crisis_mode else 10))

    # Anti-hallucinated empathy: rewrite if suspicious
    if likely_invented_hardship(out, user_text, context_summary, crisis_mode=crisis_mode):
        out = rewrite_if_needed(
            model, tokenizer, out, user_text, context_summary,
            advice_allowed=plan.advice_allowed, cache_disabled=cache_disabled
        )
        # Post-enforce again
        if not plan.advice_allowed and not crisis_mode:
            out = strip_unsolicited_advice(out)
        out = enforce_one_open_question(out, max_sents=(7 if crisis_mode else 10))

    # Decay locks AFTER using them for this turn
    decay_locks(mem)

    return out.strip()


# =========================
# 13) CLI LOOP
# =========================
def main():
    model, tokenizer, cache_disabled = load_phi3()
    mem = load_memory()

    history: List[Dict[str, str]] = []
    print("Phi-3 + EMPATHY_LEVEL_3 (type 'exit' to quit)\n")

    while True:
        user_text = builtins.input("You: ").strip()
        if user_text.lower() in ("exit", "quit"):
            save_memory(mem)
            print("Assistant: Bye!")
            break

        reply = level3_reply(model, tokenizer, history, user_text, mem, cache_disabled)
        print("\nAssistant:", reply, "\n")

        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": reply})
        history = history[-8:]  # last 4 turns

        save_memory(mem)


if __name__ == "__main__":
    main()
