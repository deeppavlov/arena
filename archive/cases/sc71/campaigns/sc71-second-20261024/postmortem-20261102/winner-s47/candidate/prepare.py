#!/usr/bin/env python3
"""prepare.py — строит FTS5/BM25 индекс KB (idx.sqlite3) в workspace_dir.

От S03 (candidate-sc71-s03-bm25) отличается двумя таблицами поиска:
1) `segments` — абзацные сегменты с локаторами (как в S03; + сегмент «документ»
   целиком для файлов <=3000 байт);
2) `docheads` — ПО ОДНОЙ СТРОКЕ на документ: «паспорт документа» из (а) головы
   файла с реквизитами, (б) всех номеров приказов/распоряжений и дат, которые
   встречаются ГДЕ УГОДНО в документе (важно для редакций: вшитый приказ-
   изменение лежит в глубине текста), (в) оглавления: локатор + начало каждого
   абзаца. Эта строка поднимает длинные «Положения»/«Регламенты» в поиск по
   названию и реквизитам без жестокого штрафа BM25 за длину, а также позволяет
   видеть сразу все редакции одного документа.

БЕЗ модельных вызовов. Вход: JSON со stdin: {assets_dir, workspace_dir, ...}.
Выход: JSON {"ok": true, "context": "..."} — контекст для решателя.
"""
import json
import re
import sqlite3
import sys
from pathlib import Path

PARA_SPLIT_RE = re.compile(r"\n\s*\n")
WS_RE = re.compile(r"\s+")
NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s")          # «5.3», «1.1.1», «4»
ROMAN_RE = re.compile(r"^([IVXLCDM]+)\.\s")            # «III. Порядок ...»
SMALL_DOC_BYTES = 3000
WORD_LOCATOR_MAX = 45

# Dochead: паспорт документа
HEAD_CHARS = 1500                 # реквизиты + первые пункты
OUTLINE_HEAD = 90                 # начало каждого абзаца в оглавление
DOCHEAD_MAX = 6500                # потолок длины строки dochead
DOCNAME_TOKENS = 4                # сколько значащих токенов в «имени» для группировки редакций

# Материализованный сигнал «тип копии» (S24): в паспорт каждого документа
# добавляется поле ТИП КОПИИ (полная копия / PDF-скан/фрагмент / фрагмент-частный),
# ОБЪЁМ (число сегментов, знаков, размер исходника) и ИСТОЧНИК (путь исходного
# файла из /assets/kb/index.tsv — данные KB). Тип вычисляется ОБЩИМ правилом
# без зашитых id/номеров: исходник не-PDF со структурными маркерами полноты и
# заметным объёмом = «полная копия»; исходник PDF = «PDF-скан/фрагмент»;
# малый объём/частный вырезка = «фрагмент/частный».
APPROVED_MARK_RE = re.compile(r"(?i)утвержд")          # УТВЕРЖДЕНО/УТВЕРЖДАЮ/утверждении
APPENDIX_MARK_RE = re.compile(r"(?i)приложен")          # приложение/приложения
TOC_MARK_RE = re.compile(r"(?i)оглавлен")               # оглавление
FRAGMENT_MIN_CHARS = 4000          # ниже — считаем «фрагментом», а не копией
FRAGMENT_MIN_SEGS = 6              # ниже число сегментов — «фрагмент»

# Паттерны реквизитов: номера приказов/распоряжений/постановлений и даты
REQ_NUM_RE = re.compile(r"(?i)(?:№|N|No)\s*[#№]?\s*(\d{1,5}(?:-\d{1,4})?)")
REQ_DATE_RE = re.compile(r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{4})\b")
APPROVE_RE = re.compile(r"(?i)об\s+утверждении\s+([^.]{0,120})")
TITLE_LINE_RE = re.compile(
    r"(?im)^\s*(ПОЛОЖЕНИЕ|РЕГЛАМЕНТ|ИНСТРУКЦИЯ|ПРАВИЛА|ПОРЯДОК|ТИПОВАЯ\s+ФОРМА|ФОРМА"
    r"|ПОЛОЖЕНИЯ|ПАМЯТКА|ПРИЛОЖЕНИЕ)\b[^\n]{0,130}")

ALNUM_RE = re.compile(r"[^\w\s]|_", re.UNICODE)


def collapse(text: str) -> str:
    return WS_RE.sub(" ", text).strip()


def norm_dir(assets_dir: str) -> Path:
    base = Path(assets_dir) / "kb" / "norm"
    if base.is_dir():
        return base
    fallback = Path("/assets/kb/norm")
    if fallback.is_dir():
        return fallback
    raise FileNotFoundError(f"norm dir not found: {base}")


def docname_of(raw: str) -> str:
    """Упрощённое «имя документа» для обнаружения редакций одного документа."""
    m = APPROVE_RE.search(raw)
    name = ""
    if m:
        name = collapse(m.group(1))
    else:
        m = TITLE_LINE_RE.search(raw)
        if m:
            name = collapse(m.group(0))
    if not name:
        name = collapse(raw[:160])
    toks = [t.lower() for t in ALNUM_RE.sub(" ", name).split() if t.strip()]
    return " ".join(toks[:DOCNAME_TOKENS])


def load_source_index(assets_dir: str) -> dict[str, tuple[str, str]]:
    """Карта origin_id -> (путь исходника, размер) из /assets/kb/index.tsv (данные KB).

    Путь исходного файла (например «документы общее скачанное/…pdf» vs
    «сценарии по ЛНА/Закупки/…docx») — легитимные данные KB, из них вычисляется
    тип копии. Не найдено записей — возвращает пустую карту (без падения)."""
    candidates = [Path(assets_dir) / "kb" / "index.tsv", Path("/assets/kb/index.tsv")]
    path = next((p for p in candidates if p.is_file()), None)
    src: dict[str, tuple[str, str]] = {}
    if path is None:
        return src
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) >= 3 and re.fullmatch(r"[0-9a-f]{64}", parts[0]):
                src[parts[0]] = (parts[2], parts[1] if len(parts) > 1 else "")
    except OSError:
        return {}
    return src


def copy_type(raw: str, segments: list, src_path: str) -> tuple[str, str]:
    """Возвращает (тип_копии, метка_объёма) общим эвристическим правилом.

    Маркеры полноты: «УТВЕРЖДЕНО/утверждаю/утверждении», «приложение»,
    «оглавление», структурные заголовки (пункты/разделы). Полная копия — это
    НЕ-PDF исходник со структурными маркерами и заметным объёмом; PDF-исходник
    (скан/печать) помечается «PDF-скан/фрагмент»; малый объём без маркеров —
    «фрагмент/частный». Никакие конкретные id/номера/названия не зашиты."""
    low = raw.lower()
    has_approve = bool(APPROVED_MARK_RE.search(low))
    has_appendix = bool(APPENDIX_MARK_RE.search(low))
    has_toc = bool(TOC_MARK_RE.search(low))
    struct_heads = 0
    for line in raw.splitlines():
        line = line.strip()
        if NUM_RE.match(line) or ROMAN_RE.match(line):
            struct_heads += 1
    seg_count = len(segments)
    txt_len = len(raw)
    content_full = (has_approve or has_appendix or has_toc or struct_heads >= 4) \
        and (seg_count >= FRAGMENT_MIN_SEGS or txt_len >= FRAGMENT_MIN_CHARS)
    src = (src_path or "").lower()
    is_pdf = src.endswith(".pdf")
    if not content_full:
        ctype = "фрагмент/частный"
    elif is_pdf:
        ctype = "PDF-скан/фрагмент"
    else:
        ctype = "полная копия"
    vol = f"{seg_count} сегм., ~{txt_len} зн."
    cmeta = f"{ctype}: {vol}"
    return ctype, cmeta


def build_dochead(raw: str, segments: list[tuple[str, str, str]],
                  src_path: str = "", src_size: str = "") -> tuple[str, str, str]:
    """Возвращает (текст паспорта, тип_копии, метка_объёма)."""
    ctype, cmeta = copy_type(raw, segments, src_path)
    parts = []
    meta = []
    if ctype:
        meta.append(f"ТИП КОПИИ: {ctype}")
    meta.append(f"ОБЪЁМ: {len(segments)} сегментов, ~{len(raw)} знаков")
    if src_path:
        meta.append(f"ИСТОЧНИК: {src_path}")
    if src_size:
        meta.append(f"РАЗМЕР ИСХОДНИКА: {src_size} б.")
    if meta:
        parts.append("МЕТА-КОПИЯ: " + " / ".join(meta))
    head = collapse(raw[:HEAD_CHARS])
    if head:
        parts.append("РЕКВИЗИТЫ: " + head)
    nums = list(dict.fromkeys(m.group(1) for m in REQ_NUM_RE.finditer(raw)))
    dates = list(dict.fromkeys(REQ_DATE_RE.findall(raw)))
    if nums:
        parts.append("ПРИКАЗЫ-РАСПОРЯЖЕНИЯ: " + " ".join(nums))
    if dates:
        parts.append("ДАТЫ: " + " ".join(dates))
    outl = []
    for _oid, loc, text in segments:
        start = collapse(text)[:OUTLINE_HEAD]
        if not start:
            continue
        tag = loc if loc else "[--]"
        outl.append(f"{tag}|{start}")
    if outl:
        parts.append("СТРУКТУРА: " + " ; ".join(outl))
    text = " ".join(parts)
    return text[:DOCHEAD_MAX], ctype, cmeta


def segment_document(path: Path) -> tuple[list[tuple[str, str, str]], str]:
    """Возвращает ([(origin_id, locator, text), ...], docname)."""
    origin_id = path.stem
    raw = path.read_text(encoding="utf-8")
    rows = []
    last_loc = ""
    for block in PARA_SPLIT_RE.split(raw):
        if not block.strip():
            continue
        first_line = block.split("\n", 1)[0].strip()
        m = NUM_RE.match(first_line) or ROMAN_RE.match(first_line)
        loc = last_loc
        if m:
            loc = m.group(1)
            last_loc = loc
        else:
            head = collapse(first_line)
            if loc == last_loc and 0 < len(head) <= WORD_LOCATOR_MAX and not head.endswith("."):
                if re.search(r"[А-Яа-яЁёA-Za-z]", head):
                    loc = head.lower()
                    last_loc = loc
        text = collapse(block)
        if text:
            rows.append((origin_id, loc, text))
    if path.stat().st_size <= SMALL_DOC_BYTES:
        rows.append((origin_id, "документ", collapse(raw)))
    return rows, docname_of(raw)


def main() -> int:
    payload = json.load(sys.stdin)
    assets = payload["assets_dir"]
    workspace = Path(payload["workspace_dir"])
    workspace.mkdir(parents=True, exist_ok=True)

    ndir = norm_dir(assets)
    src_index = load_source_index(assets)
    seg_rows = []
    head_rows = []
    files = sorted(ndir.glob("*.md"))
    for path in files:
        segs, dn = segment_document(path)
        seg_rows.extend(segs)
        raw = path.read_text(encoding="utf-8")
        src_path, src_size = src_index.get(path.stem, ("", ""))
        head_text, ctype, cmeta = build_dochead(raw, segs, src_path, src_size)
        head_rows.append((path.stem, dn, head_text, ctype, cmeta))

    db_path = workspace / "idx.sqlite3"
    con = sqlite3.connect(str(db_path))
    try:
        con.execute("PRAGMA journal_mode=OFF")
        con.execute("PRAGMA synchronous=OFF")
        con.execute("CREATE VIRTUAL TABLE segments USING fts5("
                    "origin_id UNINDEXED, locator UNINDEXED, text)")
        con.executemany(
            "INSERT INTO segments (origin_id, locator, text) VALUES (?, ?, ?)", seg_rows)
        con.execute("CREATE VIRTUAL TABLE docheads USING fts5("
                    "origin_id UNINDEXED, docname UNINDEXED, text, "
                    "ctype UNINDEXED, cmeta UNINDEXED)")
        con.executemany(
            "INSERT INTO docheads (origin_id, docname, text, ctype, cmeta) "
            "VALUES (?, ?, ?, ?, ?)", head_rows)
        con.commit()
    finally:
        con.close()

    context = (
        f"KB-индекс готов (idx.sqlite3): {len(seg_rows)} сегментов, "
        f"{len(head_rows)} паспортов-документов. Инструмент arena_search (BM25) "
        f"возвращает и фрагменты норм (text+locator — НОРМАТИВНЫЙ текст), и паспорта "
        f"документов (помечены kind=document_passport и префиксом «паспорт документа; "
        f"МЕТАДАННЫЕ — НЕ нормативный текст, не цитировать»; содержат МЕТА-КОПИЯ с "
        f"ТИП КОПИИ (полная копия / PDF-скан/фрагмент / фрагмент-частный), ОБЪЁМ, "
        f"ИСТОЧНИК — путь исходного файла из index.tsv, РЕКВИЗИТЫ/ПРИКАЗЫ-РАСПОРЯЖЕНИЯ/"
        f"ДАТЫ/СТРУКТУРА и путь /kb/norm/<id>.md). Паспорт — это служебные МЕТАДАННЫЕ "
        f"для навигации, сравнения копий и выбора действующей редакции: его текст "
        f"НЕ цитируется и из него НЕ берутся локаторы; дословная цитата — только из "
        f"самого документа /kb/norm/<id>.md. Метка ТИП КОПИИ в паспорте — это сигнал "
        f"для выбора копии одного акта (предпочти «полную копию»), а не признак "
        f"непригодности: если для акта есть только PDF-скан/фрагмент — его можно "
        f"цитировать. Если несколько документов имеют одно название — это редакции: "
        f"читайте все и применяйте действующую."
    )
    print(json.dumps({"ok": True, "context": context,
                      "segments": len(seg_rows), "docheads": len(head_rows),
                      "documents": len(files)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
