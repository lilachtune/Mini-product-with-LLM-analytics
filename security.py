import re
import logging

logger = logging.getLogger(__name__)

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+your\s+(system\s+)?prompt",
    r"forget\s+(your\s+)?(instructions|rules|guidelines)",
    r"you\s+are\s+now\s+(a|an)",
    r"act\s+as\s+(a|an|if)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"new\s+(system\s+)?prompt\s*:",
    r"(system|hidden)\s+instructions?",
    r"jailbreak",
    r"dan\s+mode",
    r"developer\s+mode",
    r"override\s+(your\s+)?(instructions|rules)",
    r"disregard\s+(your\s+)?(previous|all)",
    r"you\s+(must|should)\s+now",
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"print\s+(your\s+)?(system\s+)?prompt",
    r"what\s+(are|is)\s+your\s+(system\s+)?prompt",
    r"translate\s+your\s+(instructions|prompt)",
    r"[<\[]\s*system\s*[>\]]",
    r"\bsystem\s*:\s*(new|ignore|forget|override|you\s+are)",
    r"</?\s*instructions?\s*/?>",
    r"<\s*prompt\s*>",
]

DANGEROUS_CODE_PATTERNS = [
    r"__import__\s*\(",
    r"import\s+os",
    r"import\s+sys",
    r"import\s+subprocess",
    r"import\s+socket",
    r"import\s+urllib",
    r"import\s+requests",
    r"import\s+http",
    r"open\s*\(",
    r"exec\s*\(",
    r"eval\s*\(",
    r"compile\s*\(",
    r"__builtins__",
    r"__globals__",
    r"__class__",
    r"getattr\s*\(.+,\s*['\"]__",
    r"subprocess",
    r"shutil",
    r"pathlib",
]


def check_prompt_injection(text: str) -> tuple[bool, str]:
    if not text:
        return True, ""

    if len(text) > 2000:
        return False, "Текст слишком длинный (максимум 2000 символов)"

    text_lower = text.lower()

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            logger.warning(f"Обнаружен prompt injection. Паттерн: '{pattern}'")
            return False, f"Обнаружена попытка манипуляции инструкциями (паттерн: {pattern})"

    return True, ""


def check_code_safety(code: str) -> tuple[bool, str]:
    for pattern in DANGEROUS_CODE_PATTERNS:
        if re.search(pattern, code, re.IGNORECASE):
            return False, f"Потенциально опасная операция в коде: {pattern}"

    return True, ""


def sanitize_user_context(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r'[\x00-\x08\x0b-\x1f\x7f]', '', text)
    text = text[:1500]

    return text.strip()