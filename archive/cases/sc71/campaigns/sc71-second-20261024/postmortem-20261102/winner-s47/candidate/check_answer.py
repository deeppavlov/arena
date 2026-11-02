import json
import re
import sys
from pathlib import Path

ORIGIN = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
LINK = re.compile(r"/kb/norm/([0-9a-f]{64})\.md\b", re.IGNORECASE)
QUOTES = re.compile(r'«(.*?)»|“(.*?)”|„(.*?)“|"([^"]*)"', re.DOTALL)
UNICODE_QUOTES = re.compile(r"«(.*?)»|“(.*?)”|„(.*?)“", re.DOTALL)
KB = Path("/assets/kb/norm")


def normalized(value: str) -> str:
    return " ".join(value.split()).casefold()


def is_refusal(answer: str) -> bool:
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if not lines:
        return False
    return normalized(lines[0]).rstrip(".").strip() == "ответ отсутствует в базе"


def check_links_exist(answer: str):
    """Детерминированный гейт существования ссылок (класс nonexistent_citation,
    включая отказные перечни и цитатные утверждения): КАЖДАЯ ссылка вида
    /kb/norm/<64hex>.md в ответе должна указывать на реальный файл
    /assets/kb/norm/<id>.md. Ссылка на несуществующий документ возвращается в
    issues (ok=false) — модель обязана исправить/убрать её ДО ответа.
    Возвращает (existing_origins, issues)."""
    existing: list[str] = []
    issues: list[dict] = []
    seen: set[str] = set()
    for origin in LINK.findall(answer):
        oid = origin.lower()
        if oid in seen:
            continue
        seen.add(oid)
        path = KB / f"{oid}.md"
        if ORIGIN.fullmatch(oid) and path.is_file():
            existing.append(oid)
        else:
            issues.append({"link": f"/kb/norm/{oid}.md", "origin_id": oid,
                           "exists": False,
                           "error": "document does not exist in /assets/kb/norm",
                           "class": "nonexistent_citation"})
    return existing, issues


def json_answer(answer: str):
    text = answer.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    elif text.startswith("```\n") and text.endswith("```"):
        text = text[4:-3].strip()
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def answer_quotes(answer: str) -> list[str]:
    structured = json_answer(answer)
    found: list[str] = []

    def collect(value, key=""):
        if isinstance(value, dict):
            for name, child in value.items():
                collect(child, name)
        elif isinstance(value, list):
            for child in value:
                collect(child, key)
        elif isinstance(value, str):
            if key.casefold() in {"quote", "цитата", "citation_quote"}:
                found.append(value)
            else:
                found.extend(next(p for p in m if p) for m in UNICODE_QUOTES.findall(value) if any(m))

    if structured is not None:
        collect(structured)
    else:
        found.extend(next(p for p in m if p) for m in QUOTES.findall(answer) if any(m))
    return list(dict.fromkeys(v for v in found if len(normalized(v)) >= 20))


def main() -> None:
    payload = json.load(sys.stdin)
    arguments = payload.get("arguments") or {}
    answer = arguments.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        print(json.dumps({"ok": False, "error": "answer must be a non-empty string"}, ensure_ascii=False))
        return
    # Гейт существования ссылок применяется к любому ответу, включая отказной
    # перечень проверенных документов: несуществующая ссылка — это
    # nonexistent_citation, ответ не отправляется ни при каких условиях.
    cited, existence_issues = check_links_exist(answer)
    issues = list(existence_issues)
    # Отказной ответ: первая непустая строка — маркер «ответ отсутствует в базе».
    # Отказ не несёт цитатных утверждений и фабрикации к нему неприменимы —
    # проверка цитат (дословности) пропускается, чтобы гейт не блокировал
    # легитимный отказ (например, перечень проверенных документов в кавычках
    # без ссылок или перечень валидных документов со ссылками).
    if is_refusal(answer):
        print(json.dumps({"ok": not issues, "total_quotes": 0,
                          "checked_origins": sorted(cited),
                          "issues": issues, "refusal": True}, ensure_ascii=False))
        return
    quotes = answer_quotes(answer)
    for quote in quotes:
        cond = normalized(quote)
        found_in = []
        for origin in sorted(cited):
            path = KB / f"{origin}.md"
            if not ORIGIN.fullmatch(origin) or not path.is_file():
                continue
            if cond in normalized(path.read_text(encoding="utf-8")):
                found_in.append(origin)
        if not found_in:
            issues.append({"quote": quote, "checked_origins": sorted(cited),
                           "normalized_chars": len(cond), "found": False})
    print(json.dumps({
        "ok": not issues, "total_quotes": len(quotes),
        "checked_origins": sorted(cited), "issues": issues,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
