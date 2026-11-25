#!/usr/bin/env python3
"""Insert \\index{} tags into chapter files.

Rules
- A term is tagged at its first occurrence in each chapter.
- Where a term is defined in bold (\\textbf{term}), that occurrence is tagged with
  |textbf so the defining page prints in bold in the index; at most one per section.
- Nothing is tagged inside headings, captions, labels/refs/cites, comments, display
  math, figures, tables, TikZ, code, or the end-of-chapter Exercises.
Usage: addindex.py SRC_DIR OUT_DIR ch01.tex ch02.tex ...
"""
import re, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from terms import TERMS

SKIP_ENVS = ["equation", "equation*", "align", "align*", "gather", "gather*", "multline",
             "figure", "figure*", "table", "table*", "tikzpicture", "axis", "semilogxaxis",
             "tabular", "tabularx", "longtable", "lstlisting", "verbatim", "minted", "minipage"]
TEXT_CMDS = {"textbf", "emph", "textit", "footnote", "textsc", "underline", "mbox", "textsl"}
WORD = r"A-Za-z0-9"

def brace_end(t, i):
    """t[i] is '{' or '['; return index just past the matching close."""
    op = t[i]; cl = "}" if op == "{" else "]"
    depth = 0; j = i
    while j < len(t):
        c = t[j]
        if c == "\\": j += 2; continue
        if c == op: depth += 1
        elif c == cl:
            depth -= 1
            if depth == 0: return j + 1
        j += 1
    return len(t)

def protected_spans(t):
    hard, soft, bold = [], [], []
    # comments
    for m in re.finditer(r"(?<!\\)%[^\n]*", t): hard.append((m.start(), m.end()))
    # skipped environments (handle nesting of same name by simple search)
    for env in SKIP_ENVS:
        b = re.escape(r"\begin{%s}" % env); e = re.escape(r"\end{%s}" % env)
        for m in re.finditer(b, t):
            n = re.compile(e).search(t, m.end())
            hard.append((m.start(), n.end() if n else len(t)))
    # display math
    for m in re.finditer(r"\\\[.*?\\\]|\$\$.*?\$\$", t, re.S): hard.append((m.start(), m.end()))
    # commands and their arguments
    i = 0
    for m in re.finditer(r"\\([A-Za-z@]+\*?|.)", t):
        name = m.group(1).rstrip("*")
        if not name[:1].isalpha():
            continue
        hard.append((m.start(), m.end()))
        j = m.end()
        if name in TEXT_CMDS:
            if name == "textbf" and j < len(t) and t[j] == "{":
                bold.append((j, brace_end(t, j)))
            continue
        if not name.isalpha():
            continue
        # protect following argument groups (allow whitespace before [ or {)
        while True:
            k = j
            while k < len(t) and t[k] in " \t": k += 1
            if k < len(t) and t[k] in "{[":
                end = brace_end(t, k); hard.append((m.start(), end)); j = end
            else:
                break
    # inline math (soft)
    for m in re.finditer(r"(?<!\\)\$(?!\$)(?:[^$\\]|\\.)+?\$|\\\(.*?\\\)", t, re.S):
        soft.append((m.start(), m.end()))
    return hard, soft, bold

def merge(spans):
    spans = sorted(spans); out = []
    for a, b in spans:
        if out and a <= out[-1][1]: out[-1] = (out[-1][0], max(out[-1][1], b))
        else: out.append((a, b))
    return out

def build_patterns():
    pats = []
    for key, forms in TERMS:
        for f in forms:
            plural = f.endswith("~"); f = f.rstrip("~")
            cs = f == f.upper() and any(c.isalpha() for c in f)
            rx = re.escape(f).replace(r"\ ", r"\s+")
            rx = rx.replace("\\-\\-", "(?:--|-)")
            if plural: rx += r"(?:s|es)?"
            lead = r"(?<![%s\\])" % WORD if f[0].isalnum() else ""
            trail = r"(?![%s])" % WORD if f[-1].isalnum() or plural else ""
            flags = 0 if cs else re.I
            pats.append((len(f), key, re.compile(lead + rx + trail, flags), "$" in f))
    pats.sort(key=lambda p: -p[0])  # longest first
    return pats

PATS = build_patterns()

def sections(t):
    """Return list of section start offsets (chapter start counts as one)."""
    return [0] + [m.start() for m in re.finditer(r"\\section\*?\{", t)]

def process(t):
    ex = re.search(r"\\section\*?\{Exercises\}", t)
    limit = ex.start() if ex else len(t)
    hard, soft, bold = protected_spans(t)
    hard = merge(hard); soft = merge(soft)
    import bisect
    hs = [a for a, b in hard]
    def in_hard(a, b):
        k = bisect.bisect_right(hs, b - 1) - 1
        while k >= 0 and hard[k][1] > a:
            if hard[k][0] < b and hard[k][1] > a: return True
            k -= 1
        return False
    def soft_ok(a, b, allow):
        for x, y in soft:
            if y <= a or x >= b: continue
            if x >= a and y <= b and allow: continue
            return False
        return True
    def bold_span(a, b):
        for x, y in bold:
            if x < a and b < y: return (x, y)
        return None
    secstarts = sections(t)
    def sec_of(p): return bisect.bisect_right(secstarts, p) - 1

    taken = []  # character ranges already claimed by a match
    def overlaps(a, b): return any(not (b <= x or a >= y) for x, y in taken)
    hits = {}   # key -> list of (start, end, is_def, section)
    for _, key, rx, allow in PATS:
        for m in rx.finditer(t, 0, limit):
            a, b = m.start(), m.end()
            if in_hard(a, b) or not soft_ok(a, b, allow) or overlaps(a, b): continue
            taken.append((a, b))
            bs = bold_span(a, b)
            is_def = False
            if bs:
                inner = re.sub(r"^(the|a|an)\s+", "", t[bs[0]+1:bs[1]-1].strip(), flags=re.I).rstrip(".:,;")
                is_def = inner.lower().startswith(t[a:b].lower()) and len(inner) <= (b - a) + 14
            pos = bs[1] if is_def else b
            # do not put a tag inside soft math or right before a closing math $
            hits.setdefault(key, []).append((a, pos, is_def, sec_of(a)))
    inserts = []
    for key, hs_ in hits.items():
        hs_.sort()
        defs = {}
        for a, pos, d, s in hs_:
            if d and s not in defs: defs[s] = pos
        chosen = [(p, key + "|textbf") for p in defs.values()]
        first = hs_[0]
        def_secs = set(defs)
        if not first[2] and first[3] not in def_secs and (not defs or first[3] < min(def_secs)):
            chosen.append((first[1], key))
        if not defs and not chosen:
            chosen.append((first[1], key))
        # de-duplicate positions
        for p, k in chosen: inserts.append((p, k))
    # apply
    inserts = sorted(set(inserts), key=lambda x: (-x[0], x[1]))
    for p, k in inserts:
        t = t[:p] + r"\index{%s}" % k + t[p:]
    return t, len(inserts)

if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    total = 0
    for f in sys.argv[3:]:
        t = open(os.path.join(src, f), encoding="utf-8").read()
        if "\\index{" in t:
            print(f, "already has \\index tags; skipped"); continue
        nt, n = process(t)
        open(os.path.join(out, f), "w", encoding="utf-8").write(nt)
        print(f"{f}: {n} tags"); total += n
    print("total", total)
