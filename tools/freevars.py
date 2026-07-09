#!/usr/bin/env python3
"""R3-static spike: enumerate the free (primitive) identifiers referenced by a
flattened linklet such as expander/expander.rktl.

The production C++ parser cannot yet ingest expander.rktl (rational/flonum and
`1/foo` identifier lexing are M3 work), so this standalone, tolerant,
scope-aware s-expression walker produces the primitive worklist that seeds
M4-M7. It performs a real free-variable analysis: it tracks lexical scope for
the FEP binding forms (lambda, case-lambda, let-values, letrec-values,
define-values), skips quoted data, and reports every identifier referenced but
neither locally bound nor defined by the linklet, ranked by frequency.

Usage: python3 tools/freevars.py expander/expander.rktl [out.tsv]
"""
import re
import sys
from collections import Counter

# ---------------------------------------------------------------------------
# Tolerant reader: produces a nested tree. Nodes are:
#   list            -> a Python list of child nodes
#   ("S", name)     -> a symbol (identifier)
#   ("A",)          -> any non-symbol atom (number/string/char/bool/kw/bytes)
# Reader-shorthand and self-quoting literals are wrapped as ["quote", ...] or
# tagged data so the walker skips them.
# ---------------------------------------------------------------------------

OPEN = set("([{")
CLOSE = set(")]}")
DELIM = set(" \t\n\r\f\v()[]{}\"';`,")

NUM_RE = re.compile(
    r"""^[+-]?(
        \d+ |                      # integer
        \d+/\d+ |                  # rational
        (\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)? |  # decimal / float
        (inf|nan)\.[0f]            # +inf.0 -nan.0 etc (sign required, handled above)
    )$""",
    re.VERBOSE,
)


def is_number(tok: str) -> bool:
    if NUM_RE.match(tok):
        return True
    if tok in ("+inf.0", "-inf.0", "+nan.0", "-nan.0", "+inf.f", "-inf.f",
               "+nan.f", "-nan.f"):
        return True
    return False


def read_tree(s: str):
    """Parse the whole string into a root list of top-level datums."""
    n = len(s)
    i = 0
    root = []
    stack = [root]            # stack[-1] is the current list being filled
    wrap = [0]                # pending quote-wraps for the next datum at this level
    childwrap = []            # wrap to apply to a child list when it closes

    def add(datum):
        w = wrap[-1]
        wrap[-1] = 0
        for _ in range(w):
            datum = [("S", "quote"), datum]
        stack[-1].append(datum)

    while i < n:
        c = s[i]
        if c in " \t\n\r\f\v":
            i += 1
            continue
        if c == ";":  # line comment
            j = s.find("\n", i)
            i = n if j < 0 else j + 1
            continue
        if c in OPEN:
            childwrap.append(wrap[-1]); wrap[-1] = 0
            stack.append([]); wrap.append(0)
            i += 1
            continue
        if c in CLOSE:
            lst = stack.pop(); wrap.pop()
            w = childwrap.pop() if childwrap else 0
            for _ in range(w):
                lst = [("S", "quote"), lst]
            stack[-1].append(lst)
            i += 1
            continue
        if c == '"':                       # string
            i = skip_string(s, i + 1, n); add(("A",)); continue
        if c == "'":                       # quote
            wrap[-1] += 1; i += 1; continue
        if c == "`":                       # quasiquote -> treat as data, skip
            wrap[-1] += 1; i += 1; continue
        if c == ",":                       # unquote / unquote-splicing -> data
            i += 2 if s[i:i + 2] == ",@" else 1
            wrap[-1] += 1; continue
        if c == "#":
            i = read_hash(s, i, n, stack, wrap, childwrap, add)
            continue
        # default: a constituent token (symbol or number)
        j = i
        while j < n and s[j] not in DELIM:
            j += 1
        tok = s[i:j]; i = j
        add(("A",) if is_number(tok) else ("S", tok))
    return root


def skip_string(s, i, n):
    while i < n:
        c = s[i]
        if c == "\\":
            i += 2; continue
        if c == '"':
            return i + 1
        i += 1
    return n


def read_hash(s, i, n, stack, wrap, childwrap, add):
    """Handle a token starting with '#'. Returns the new index."""
    c1 = s[i + 1] if i + 1 < n else ""
    if c1 == "|":                                  # block comment (nestable)
        depth = 1; j = i + 2
        while j < n and depth:
            two = s[j:j + 2]
            if two == "#|":
                depth += 1; j += 2
            elif two == "|#":
                depth -= 1; j += 2
            else:
                j += 1
        return j
    if c1 == ";":                                  # datum comment: drop next datum
        # emit a throwaway wrap that we cancel by adding to a scratch list
        # (none present in expander.rktl; handled defensively)
        wrap[-1] += 0
        return i + 2  # best-effort: skip the '#;' marker only
    if c1 == "\\":                                 # char literal
        j = i + 2
        if j < n and (s[j].isalpha()):
            k = j
            while k < n and (s[k].isalnum() or s[k] == "-"):
                k += 1
            j = k if k > j + 1 else j + 1          # named char, else single char
        else:
            j = j + 1 if j < n else j
        add(("A",)); return j
    if c1 in OPEN:                                 # #( vector literal -> data
        childwrap.append(wrap[-1] + 1); wrap[-1] = 0  # +1 so it is skipped as data
        stack.append([]); wrap.append(0)
        return i + 2
    if c1 == '"':                                  # #"..." byte string
        j = skip_string(s, i + 2, n); add(("A",)); return j
    if c1 == "%":                                  # #%... symbol (identifier)
        j = i
        while j < n and s[j] not in DELIM:
            j += 1
        add(("S", s[i:j])); return j
    if c1 == ":":                                  # #:keyword atom
        j = i
        while j < n and s[j] not in DELIM:
            j += 1
        add(("A",)); return j
    # #hash( / #hasheq( / #s( / #N( ... -> data-open; else #t/#f/#x.. atom
    j = i + 1
    while j < n and (s[j].isalnum()):
        j += 1
    if j < n and s[j] in OPEN:                      # #word( -> data literal
        childwrap.append(wrap[-1] + 1); wrap[-1] = 0
        stack.append([]); wrap.append(0)
        return j + 1
    # plain #-atom (boolean / radix number / etc.)
    k = i
    while k < n and s[k] not in DELIM:
        k += 1
    add(("A",)); return k


# ---------------------------------------------------------------------------
# Free-variable analysis
# ---------------------------------------------------------------------------

SKIP_HEADS = {"quote", "quote-syntax", "quasiquote", "unquote",
              "unquote-splicing"}
SEQ_HEADS = {"if", "begin", "begin0", "with-continuation-mark", "#%expression",
             "#%app", "#%plain-app", "begin-for-syntax"}
IGNORE_REFS = {"."}


def sym(node):
    return node[1] if isinstance(node, tuple) and node[0] == "S" else None


def formals_set(node):
    """Bound identifiers of a lambda/case-lambda formal spec."""
    out = set()
    s = sym(node)
    if s is not None:                    # rest-arg: (lambda x ...)
        out.add(s)
        return out
    if isinstance(node, list):           # (a b . rest) / (a b) / ()
        for e in node:
            es = sym(e)
            if es is not None and es != ".":
                out.add(es)
    return out


def collect_defined(body):
    """First pass: every identifier bound by a define-values, ignoring quoted
    data. These are the linklet's own module-level bindings."""
    G = set()
    stack = list(body)
    while stack:
        node = stack.pop()
        if not isinstance(node, list) or not node:
            continue
        h = sym(node[0])
        if h in SKIP_HEADS:
            continue
        if h in ("define-values", "define-syntaxes") and len(node) >= 2:
            ids = node[1]
            if isinstance(ids, list):
                for e in ids:
                    es = sym(e)
                    if es:
                        G.add(es)
            else:
                es = sym(ids)
                if es:
                    G.add(es)
        stack.extend(node)
    return G


def analyze(body, G):
    """Iterative scope-aware walk. Returns Counter of free identifiers."""
    free = Counter()
    # work items: (node, scopes) where scopes is a tuple of frozenset
    work = [(f, ()) for f in reversed(body)]
    while work:
        node, scopes = work.pop()
        # atom
        if isinstance(node, tuple):
            if node[0] == "S":
                name = node[1]
                if name in IGNORE_REFS:
                    continue
                if name in G:
                    continue
                if any(name in fr for fr in scopes):
                    continue
                free[name] += 1
            continue
        if not isinstance(node, list) or not node:
            continue
        h = sym(node[0])
        if h in SKIP_HEADS:
            continue
        if h in ("lambda", "#%plain-lambda") and len(node) >= 2:
            b = frozenset(formals_set(node[1]))
            ns = scopes + (b,)
            for e in node[2:]:
                work.append((e, ns))
            continue
        if h == "case-lambda":
            for clause in node[1:]:
                if isinstance(clause, list) and clause:
                    b = frozenset(formals_set(clause[0]))
                    ns = scopes + (b,)
                    for e in clause[1:]:
                        work.append((e, ns))
            continue
        if h in ("let-values", "letrec-values", "let*-values") and len(node) >= 2:
            binds = node[1] if isinstance(node[1], list) else []
            ids = set()
            for bnd in binds:
                if isinstance(bnd, list) and bnd:
                    for e in (bnd[0] if isinstance(bnd[0], list) else [bnd[0]]):
                        es = sym(e)
                        if es and es != ".":
                            ids.add(es)
            bset = frozenset(ids)
            rec = h != "let-values"
            rhs_scope = scopes + (bset,) if rec else scopes
            body_scope = scopes + (bset,)
            for bnd in binds:
                if isinstance(bnd, list) and len(bnd) >= 2:
                    work.append((bnd[1], rhs_scope))
            for e in node[2:]:
                work.append((e, body_scope))
            continue
        if h in ("define-values", "define-syntaxes") and len(node) >= 3:
            work.append((node[2], scopes))       # ids already in G
            continue
        if h == "set!" and len(node) >= 3:
            work.append((node[1], scopes))        # target is a reference too
            work.append((node[2], scopes))
            continue
        if h == "#%variable-reference":
            for e in node[1:]:
                work.append((e, scopes))
            continue
        if h in SEQ_HEADS:
            for e in node[1:]:
                work.append((e, scopes))
            continue
        # application (operator + operands are all expressions)
        for e in node:
            work.append((e, scopes))
    return free


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "expander/expander.rktl"
    out = sys.argv[2] if len(sys.argv) > 2 else None
    s = open(path, encoding="utf-8", errors="replace").read()
    root = read_tree(s)
    linklet = None
    for d in root:
        if isinstance(d, list) and d and sym(d[0]) == "linklet":
            linklet = d
            break
    if linklet is None:
        print("error: no (linklet ...) form found", file=sys.stderr)
        sys.exit(1)
    imports, exports = linklet[1], linklet[2]
    body = linklet[3:]
    G = collect_defined(body)
    free = analyze(body, G)

    n_exports = len(exports) if isinstance(exports, list) else 0
    print(f"file:                 {path}")
    print(f"top-level body forms: {len(body)}")
    print(f"exports:              {n_exports}")
    print(f"linklet-defined ids:  {len(G)}")
    print(f"distinct FREE ids:    {len(free)}   (total refs: {sum(free.values())})")
    print()
    print("Top 60 free identifiers by reference count:")
    for name, c in free.most_common(60):
        print(f"  {c:6d}  {name}")

    if out:
        with open(out, "w") as fh:
            for name, c in free.most_common():
                fh.write(f"{c}\t{name}\n")
        print(f"\nfull ranked list -> {out}")


if __name__ == "__main__":
    main()
