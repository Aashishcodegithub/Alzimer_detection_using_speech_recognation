from typing import Optional
from .config import get_gemini_model
from .prompts import ALZ_SYSTEM_PROMPT

ALZ_KEYWORDS = [
    "alzheimer", "alzheimers", "dementia", "memory",
    "forget", "cognitive", "brain", "neurologist", "caregiver"
]


def is_alzheimer_question(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ALZ_KEYWORDS)


def build_context_text(
    class_name: Optional[str],
    probability: Optional[float],
    model_name: Optional[str],
) -> str:
    if class_name is None or probability is None:
        return (
            "No screening result is available. "
            "Provide general educational information only."
        )

    return (
        f"Screening result: {class_name}.\n"
        f"Confidence score: {probability * 100:.1f}%.\n"
        f"Model used: {model_name or 'AI speech model'}.\n"
        "This result is from an AI screening system and is NOT a medical diagnosis."
    )


def _safe_extract_text(response) -> Optional[str]:
    """
    Gemini sometimes returns HTTP 200 with empty content due to safety filtering.
    """
    try:
        if hasattr(response, "text") and response.text:
            return response.text.strip()
    except Exception:
        pass
    return None


def generate_alz_answer(
    question: str,
    class_name: Optional[str] = None,
    probability: Optional[float] = None,
    model_name: Optional[str] = None,
) -> str:

    model = get_gemini_model()
    context_text = build_context_text(class_name, probability, model_name)

    prompt = f"""
{ALZ_SYSTEM_PROMPT}

Screening context:
{context_text}

User question:
{question}

Remember:
- Educational explanation only
- No diagnosis or treatment
"""

    response = model.generate_content(prompt)
    text = _safe_extract_text(response)

    if not text:
        # 🔁 FALLBACK (IMPORTANT)
        return (
            "This AI screening result indicates patterns in speech that may be "
            "associated with cognitive changes. The confidence score reflects how "
            "strongly the model detected these patterns based on its training data. "
            "This is not a medical diagnosis, and only a qualified neurologist can "
            "provide a definitive clinical assessment."
        )

    return text
