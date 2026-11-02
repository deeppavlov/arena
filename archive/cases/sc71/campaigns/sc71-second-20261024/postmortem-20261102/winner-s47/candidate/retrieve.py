#!/usr/bin/env python3
"""retrieve.py — гибридный поиск по FTS5-индексу candidate-sc71-s07.

Двухтабличная выдача (память режима):
1) `segments` — сильные абзацные фрагменты с локаторами (как в S03);
2) `docheads` — «паспорта документов»: для запросов по названию/реквизитам
   (длинные Положения/Регламенты не тонут за штрафом BM25 по длине).

Мерж: у каждого документа лучшая оценка = max(лучший сегмент, 0.62*паспорт).
Затем: если несколько документов имеют одно «имя» (одна редакция), включаются
ВСЕ такие документы (до +2) — чтобы агент видел редакции целиком и выбрал
действующую.

Паспорт документа помечается как МЕТАДАННЫЕ: у результата проставляется
kind="document_passport", в text — префикс «паспорт документа; МЕТАДАННЫЕ — НЕ
нормативный текст, не цитировать», locator — «метаданные (паспорт) документа —
не цитируется». source остаётся /kb/norm/<id>.md (файл, к которому относится
паспорт). Паспорт служит только для навигации (какой документ, какая редакция,
какие приказы/даты) — его текст НЕ является нормативным и не должен попадать в
ответ в виде цитат или локаторов; цитаты берутся из самого документа.

Вход: JSON со stdin: {workspace_dir, arguments: {"query": ...}}.
Выход: JSON {"results": [{"text":.., "source":.., "locator":.., "kind":..}]}.
"""
import json
import re
import sqlite3
import sys
from pathlib import Path

WS_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
MAX_OR_TERMS = 30

TOP_SEG_FETCH = 70
TOP_HEAD_FETCH = 45
MAX_DOCS = 10
EXPAND_MAX = 2                # доп. редакции сверх MAX_DOCS
MAX_TEXT = 1800               # на один фрагмент-сегмент
MAX_SEGS_PER_DOC = 2
SEG2_RATIO = 0.7              # второй сегмент доки только если >=70% лучшего
HEAD_W = 0.62                 # вклад паспорта документа (норм. оценка)
DOCHEAD_SNIP = 750            # показ паспорта для head-only/доп.редакций
TOTAL_CAP = 48000             # потолок знаков на один вызов


def build_expr(query: str) -> str:
    tokens = TOKEN_RE.findall(query.lower())
    kept = []
    for tok in tokens:
        if tok.isdigit():
            if len(tok) >= 3:  # одиночные/парные цифры слишком шумны (п. 5.3, 22:00)
                kept.append(tok)
        elif len(tok) >= 2:
            kept.append(tok)
    kept = kept[:MAX_OR_TERMS]
    if not kept:
        return '"%s"' % query.replace('"', '""') if query else ""
    return " OR ".join('"%s"' % tok for tok in kept)


def main() -> int:
    payload = json.load(sys.stdin)
    workspace = Path(payload["workspace_dir"])
    query = (payload.get("arguments") or {}).get("query", "")
    results: list[dict] = []
    db_path = workspace / "idx.sqlite3"
    query = WS_RE.sub(" ", query).strip()
    if query and db_path.is_file():
        expr = build_expr(query)
        if expr:
            con = sqlite3.connect(str(db_path))
            try:
                try:
                    seg_rows = con.execute(
                        "SELECT text, origin_id, locator, bm25(segments) AS sc "
                        "FROM segments WHERE segments MATCH ? ORDER BY sc LIMIT ?",
                        (expr, TOP_SEG_FETCH)).fetchall()
                except sqlite3.OperationalError:
                    seg_rows = []
                try:
                    head_rows = con.execute(
                        "SELECT text, origin_id, docname, ctype, cmeta, "
                        "bm25(docheads) AS sc "
                        "FROM docheads WHERE docheads MATCH ? ORDER BY sc LIMIT ?",
                        (expr, TOP_HEAD_FETCH)).fetchall()
                except sqlite3.OperationalError:
                    head_rows = []
            finally:
                con.close()

            # агрегация по документам
            docs: dict[str, dict] = {}
            for text, oid, loc, sc in seg_rows:
                d = docs.setdefault(oid, {"segs": [], "head": None, "docname": "",
                                          "ctype": "", "cmeta": ""})
                d["segs"].append((loc, text, sc))
            for text, oid, dn, ctype, cmeta, sc in head_rows:
                d = docs.setdefault(oid, {"segs": [], "head": None, "docname": "",
                                          "ctype": "", "cmeta": ""})
                if d["head"] is None or sc < d["head"][0]:   # меньше = лучше (bm25)
                    d["head"] = (sc, text)
                if not d["docname"]:
                    d["docname"] = dn
                if not d["ctype"]:
                    d["ctype"] = ctype
                if not d["cmeta"]:
                    d["cmeta"] = cmeta
            # нормированные в [0,1] оценки по каждой таблице отдельно (1 = лучший)
            def norm_range(rows_with_score, idx):
                vals = [r[idx] for r in rows_with_score]
                lo = min(vals); hi = max(vals)
                span = (hi - lo) or 1.0
                return lambda v: (hi - v) / span   # bm25: меньше=лучше -> больший = лучший
            if seg_rows:
                fseg = norm_range(seg_rows, 3)
            if head_rows:
                fhead = norm_range(head_rows, 5)   # sc = 6-я колонка (после text, oid, docname, ctype, cmeta)
            # дозагрузка docname и dochead-text для ВСЕХ кандидатов
            # (нужно для name-boost ниже)
            all_oids = list(docs)
            if all_oids:
                in_list = ",".join(f'"{o}"' for o in all_oids)
                con = sqlite3.connect(str(db_path))
                try:
                    for oid, dn, ht in con.execute(
                            "SELECT origin_id, docname, text FROM docheads "
                            f"WHERE origin_id IN ({in_list})").fetchall():
                        d = docs.get(oid)
                        if d is None:
                            continue
                        if not d["docname"]:
                            d["docname"] = dn
                        if d["head"] is None:
                            d["head"] = (0.0, ht)
                    # дозагрузка метаданных копии (тип/объём) для всех кандидатов
                    for oid, ctype, cmeta in con.execute(
                            "SELECT origin_id, ctype, cmeta FROM docheads "
                            f"WHERE origin_id IN ({in_list})").fetchall():
                        d = docs.get(oid)
                        if d is None:
                            continue
                        if not d["ctype"]:
                            d["ctype"] = ctype
                        if not d["cmeta"]:
                            d["cmeta"] = cmeta
                finally:
                    con.close()
            # ----------------------------------------------------------------
            # NAME-BOOST (S19): если название/реквизиты документа содержатся в
            # запросе как НЕПРЕРЫВНАЯ фраза — документ релевантен по названию
            # даже при слабом BM25 (длинные «Положения»/«Регламенты» тонут за
            # штрафом по длине). Учитываем и docname (краткое имя), и dochead
            # (реквизиты шапки): у ca645396 docname тривиален («положение»), но
            # dochead содержит «Положение о закупке товаров, работ, услуг…».
            # Фраза строится по СЫРЫМ токенам запроса (включая «о», «по»), чтобы
            # «положение о закупке …» ложилось непрерывно на такие заголовки.
            # ----------------------------------------------------------------
            raw_toks = [t.lower() for t in TOKEN_RE.findall(query.lower())]
            kept_toks = [t for t in raw_toks
                         if (t.isdigit() and len(t) >= 3) or (not t.isdigit() and len(t) >= 2)]
            if raw_toks:
                def phrase_best(text: str, mintok: int = 2) -> float:
                    """Доля токенов запроса, покрытых самой длинной непрерывной
                    фразой, встречающейся в text (по токенам обеих сторон)."""
                    dtoks = text.split()
                    if not dtoks:
                        return 0.0
                    joined = " ".join(dtoks)
                    n = len(raw_toks)
                    for size in range(n, mintok - 1, -1):
                        for i in range(n - size + 1):
                            phrase = " ".join(raw_toks[i:i + size])
                            if phrase in joined:
                                return size / n
                    return 0.0

                def name_boost(docname: str, head_text: str) -> float:
                    boost = phrase_best(docname, mintok=3)
                    if boost < 0.75 and head_text:
                        boost = max(boost, 0.93 * phrase_best(head_text, mintok=3))
                    if not boost:
                        if kept_toks:
                            cov = sum(1 for t in kept_toks
                                      if f" {t} " in f" {docname} ")
                            if cov >= 2:
                                boost = 0.5 * (cov / len(kept_toks))
                    return boost
            for d in docs.values():
                seg_best = min((s[2] for s in d["segs"]), default=None)  # меньше=лучше
                seg_n = fseg(seg_best) if seg_best is not None and seg_rows else None
                head_n = fhead(d["head"][0]) if d["head"] is not None and head_rows else None
                cand = []
                if seg_n is not None:
                    cand.append(seg_n)
                if head_n is not None:
                    cand.append(HEAD_W * head_n)
                d["score"] = max(cand) if cand else 0.0
                if raw_toks:
                    boost = name_boost(d["docname"],
                                       d["head"][1] if d["head"] is not None else "")
                    if boost:
                        raw = d["score"]
                        d["score"] = raw + boost * (1.0 - raw)
            ranked = sorted(docs, key=lambda o: docs[o]["score"], reverse=True)
            # группы по «имени документа» (редакции)
            by_name: dict[str, list[str]] = {}
            for o in ranked:
                nm = docs[o]["docname"]
                if nm:
                    by_name.setdefault(nm, []).append(o)

            selected: list[str] = []
            for o in ranked:
                if o not in selected:
                    selected.append(o)
                if len(selected) >= MAX_DOCS:
                    break
            # добавить редакции (одинаковое имя с уже выбранным)
            added = 0
            for o in list(selected):
                nm = docs[o]["docname"]
                if not nm:
                    continue
                for sib in by_name.get(nm, []):
                    if sib not in selected:
                        selected.append(sib)
                        added += 1
                        if added >= EXPAND_MAX:
                            break
                if added >= EXPAND_MAX:
                    break

            # группы редакций среди выбранных (одинаковое «имя документа»);
            # дозагружаем паспорт для доков, попавших только через сегменты
            oids = ",".join(f'"{o}"' for o in selected)
            if oids:
                con = sqlite3.connect(str(db_path))
                try:
                    for oid, dn, ht in con.execute(
                            "SELECT origin_id, docname, text FROM docheads "
                            f"WHERE origin_id IN ({oids})").fetchall():
                        d = docs.get(oid)
                        if d is None:
                            continue
                        if not d["docname"]:
                            d["docname"] = dn
                        if d["head"] is None:
                            d["head"] = (0.0, ht)
                    for oid, ctype, cmeta in con.execute(
                            "SELECT origin_id, ctype, cmeta FROM docheads "
                            f"WHERE origin_id IN ({oids})").fetchall():
                        d = docs.get(oid)
                        if d is None:
                            continue
                        if not d["ctype"]:
                            d["ctype"] = ctype
                        if not d["cmeta"]:
                            d["cmeta"] = cmeta
                finally:
                    con.close()
            grp: dict[str, list[str]] = {}
            for o in selected:
                nm = docs[o]["docname"]
                if nm:
                    grp.setdefault(nm, []).append(o)
            multi_edition = {o for members in grp.values() if len(members) >= 2
                             for o in members}

            # рендер
            total_chars = 0
            for o in selected:
                d = docs[o]
                segs = sorted(d["segs"], key=lambda s: s[2], reverse=True)
                wants_head = (not segs) or (o in multi_edition)
                if wants_head and d["head"] is not None:
                    copy_label = d.get("cmeta") or d.get("ctype") or ""
                    label_txt = f"[{copy_label}] " if copy_label else ""
                    results.append({
                        "text": "[паспорт документа; МЕТАДАННЫЕ — НЕ нормативный текст, "
                                "не цитировать] " + label_txt + d["head"][1][:DOCHEAD_SNIP],
                        "source": f"/kb/norm/{o}.md",
                        "locator": "метаданные (паспорт) документа — не цитируется",
                        "kind": "document_passport",
                    })
                    total_chars += DOCHEAD_SNIP
                for n, (loc, text, _sc) in enumerate(segs):
                    if n >= MAX_SEGS_PER_DOC:
                        break
                    if n > 0 and _sc < segs[0][2] * SEG2_RATIO:
                        break
                    results.append({"text": text[:MAX_TEXT],
                                    "source": f"/kb/norm/{o}.md",
                                    "locator": loc})
                    total_chars += min(len(text), MAX_TEXT)
                if total_chars > TOTAL_CAP:
                    break
    print(json.dumps({"results": results}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
