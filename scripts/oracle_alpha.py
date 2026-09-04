#!/usr/bin/env python3
"""oracle_alpha.py — the binder-driven alpha-renaming walker over the
fully-expanded-program grammar (issue #92).

`normalize(datum)` renames every bound identifier in a `(module ...)` datum
to `v0, v1, ...` in first-binder-appearance order, threading one counter
across the whole traversal so structurally-identical programs (module-path
encodings and gensym counter offsets aside) always print identically —
"global-consistent", not local, alpha-renaming.
"""
from oracle_datum import DottedList, Symbol


class _Counter:
    def __init__(self):
        self.n = 0

    def fresh(self):
        name = f'v{self.n}'
        self.n += 1
        return name


def _head_name(datum):
    if isinstance(datum, list) and datum and isinstance(datum[0], Symbol):
        return datum[0].name
    return None


def _bind_formals(formals, env, ctr):
    """Assigns each symbol in a lambda/case-lambda/let-values formals shape
    (bare rest symbol, proper list, or dotted/improper list) a fresh vN,
    left to right, including a dotted tail last. Returns the renamed
    formals shape and the env extended with the new bindings."""
    new_env = dict(env)

    def bind_one(sym):
        name = ctr.fresh()
        new_env[sym.name] = name
        return Symbol(name)

    if isinstance(formals, Symbol):
        return bind_one(formals), new_env
    if isinstance(formals, list):
        return [bind_one(s) for s in formals], new_env
    if isinstance(formals, DottedList):
        new_items = [bind_one(s) for s in formals.items]
        new_tail = bind_one(formals.tail)
        return DottedList(new_items, new_tail), new_env
    raise ValueError(f'unrecognized formals shape: {formals!r}')


def _walk_module(datum, env, ctr):
    # (module name lang body...) — name and lang are inert labels/module
    # paths, never renamed. A (sub)module never sees its enclosing module's
    # bindings, so it always starts from a fresh, empty env; the shared
    # counter keeps running so numbering stays global-consistent.
    head, name, lang, *body = datum
    new_body = [_walk(form, {}, ctr) for form in body]
    return [head, name, lang] + new_body


def _collect_provided_names(forms):
    """The set of internal names a direct-child #%provide form exports —
    both the plain shape ((#%provide name ...)) and the rename shape
    ((#%provide (rename internal external) ...)), where `internal` is the
    define-values LHS being exported and `external` is just a public label,
    never a binder itself."""
    names = set()
    for form in forms:
        if _head_name(form) != '#%provide':
            continue
        for clause in form[1:]:
            if isinstance(clause, Symbol):
                names.add(clause.name)
            elif _head_name(clause) == 'rename' and len(clause) == 3:
                names.add(clause[1].name)
    return names


def _walk_module_begin(datum, env, ctr):
    # (#%module-begin form...) — every direct-child define-values LHS is in
    # scope for the *entire* body, including forms that precede it, so all
    # LHS names are assigned a vN in body order before anything is walked.
    # A name that's also exported via #%provide keeps its original text
    # (mapped to itself) instead of getting a fresh vN, both at the
    # define-values site and wherever #%provide references it — #%provide/
    # #%require need no dedicated walker beyond this: their contents are
    # otherwise never bound in env, so the generic fallback already leaves
    # them untouched.
    head, *forms = datum
    provided = _collect_provided_names(forms)
    body_env = dict(env)
    for form in forms:
        if _head_name(form) == 'define-values':
            for sym in form[1]:
                body_env[sym.name] = sym.name if sym.name in provided else ctr.fresh()

    new_forms = []
    for form in forms:
        if _head_name(form) == 'define-values':
            _, lhs, rhs = form
            new_lhs = [Symbol(body_env[sym.name]) for sym in lhs]
            new_forms.append([form[0], new_lhs, _walk(rhs, body_env, ctr)])
        else:
            new_forms.append(_walk(form, body_env, ctr))
    return [head] + new_forms


def _walk_phase_shift(datum, env, ctr):
    # define-syntaxes / begin-for-syntax introduce phase-1 code: a separate
    # scope that must not resolve free references through the phase-0 env
    # (or vice versa), even though the shared counter keeps running. Real
    # `expand` output always includes at least a define-syntaxes wherever a
    # macro is defined, so this is needed from the first fixture driven by
    # actual Racket output (slice 8), not just for hand-crafted test data.
    head, *rest = datum
    return [head] + [_walk(form, {}, ctr) for form in rest]


def _walk_lambda(datum, env, ctr):
    head, formals, *body = datum
    new_formals, body_env = _bind_formals(formals, env, ctr)
    new_body = [_walk(form, body_env, ctr) for form in body]
    return [head, new_formals] + new_body


def _walk_case_lambda(datum, env, ctr):
    # (case-lambda (formals body...) ...) — each clause's parameter list is
    # its own binder scope; clause N's formals are invisible to clause M.
    head, *clauses = datum
    new_clauses = []
    for clause in clauses:
        formals, *body = clause
        new_formals, clause_env = _bind_formals(formals, env, ctr)
        new_body = [_walk(form, clause_env, ctr) for form in body]
        new_clauses.append([new_formals] + new_body)
    return [head] + new_clauses


def _walk_let_values(datum, env, ctr):
    # (let-values ((formals rhs) ...) body...) — every rhs is evaluated in
    # the *outer* env (no clause sees another clause's, or its own,
    # bindings); only the body sees the accumulated bindings.
    head, clauses, *body = datum
    extended = dict(env)
    new_clauses = []
    for formals, rhs in clauses:
        new_rhs = _walk(rhs, env, ctr)
        new_formals, extended = _bind_formals(formals, extended, ctr)
        new_clauses.append([new_formals, new_rhs])
    new_body = [_walk(form, extended, ctr) for form in body]
    return [head, new_clauses] + new_body


def _walk_letrec_values(datum, env, ctr):
    # (letrec-values ((formals rhs) ...) body...) — all clause LHS are
    # bound before any rhs is walked, so clauses (and the body) can
    # forward-reference each other, just like #%module-begin's defines.
    head, clauses, *body = datum
    extended = dict(env)
    new_formals_list = []
    for formals, _rhs in clauses:
        new_formals, extended = _bind_formals(formals, extended, ctr)
        new_formals_list.append(new_formals)
    new_clauses = [
        [new_formals, _walk(rhs, extended, ctr)]
        for (_, rhs), new_formals in zip(clauses, new_formals_list)
    ]
    new_body = [_walk(form, extended, ctr) for form in body]
    return [head, new_clauses] + new_body


def _walk_quote(datum, env, ctr):
    # Quoted data is literal, not code: never walked/renamed here. A later
    # pass (issue #92 slice 7) handles gensym renumbering within a
    # quote/quote-syntax payload; nothing else may touch it.
    return datum


# Forms needing behavior other than "leave the head alone, walk every other
# subform in the current env" — the generic fallback in _walk below, which
# already covers #%app, if, set!, begin, begin0, with-continuation-mark,
# #%top, #%variable-reference and #%expression (none of these introduce a
# binder; set!'s target is an ordinary reference, not a new binding).
_SPECIAL_FORMS = {
    'module': _walk_module,
    '#%module-begin': _walk_module_begin,
    'define-syntaxes': _walk_phase_shift,
    'begin-for-syntax': _walk_phase_shift,
    'lambda': _walk_lambda,
    'case-lambda': _walk_case_lambda,
    'let-values': _walk_let_values,
    'letrec-values': _walk_letrec_values,
    'quote': _walk_quote,
    'quote-syntax': _walk_quote,
}


def _walk(datum, env, ctr):
    if isinstance(datum, Symbol):
        return Symbol(env.get(datum.name, datum.name))
    if isinstance(datum, list):
        if not datum:
            return datum
        head = datum[0]
        if isinstance(head, Symbol) and head.name in _SPECIAL_FORMS:
            return _SPECIAL_FORMS[head.name](datum, env, ctr)
        if isinstance(head, Symbol):
            return [head] + [_walk(e, env, ctr) for e in datum[1:]]
        return [_walk(e, env, ctr) for e in datum]
    # Numbers, booleans, strings, chars, DottedList, Vector: inert outside
    # quoted data (which _walk_quote already keeps out of reach here).
    return datum


def normalize(datum):
    if _head_name(datum) != 'module':
        raise ValueError('oracle_alpha.normalize expects a (module ...) datum')
    return _walk_module(datum, {}, _Counter())
