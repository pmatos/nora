#!/usr/bin/env python3
"""oracle_datum.py — parser/writer for the Racket `write` output subset the
oracle harness needs: symbols, numbers, booleans, strings, chars,
proper/dotted pairs, and vectors.

Plain `write` (what oracle-expand.rkt/oracle-eval.rkt use) never emits
reader shorthand (`'`/`` ` ``/`,`/`,@`) — `quote` and friends always appear
as ordinary list forms — so this parser does not support that syntax.
"""


class Symbol:
    __slots__ = ('name',)

    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return isinstance(other, Symbol) and self.name == other.name

    def __hash__(self):
        return hash(('Symbol', self.name))

    def __repr__(self):
        return f'Symbol({self.name!r})'


class Char:
    __slots__ = ('ch',)

    def __init__(self, ch):
        self.ch = ch

    def __eq__(self, other):
        return isinstance(other, Char) and self.ch == other.ch

    def __hash__(self):
        return hash(('Char', self.ch))

    def __repr__(self):
        return f'Char({self.ch!r})'


class Vector:
    __slots__ = ('items',)

    def __init__(self, items):
        self.items = items

    def __eq__(self, other):
        return isinstance(other, Vector) and self.items == other.items

    def __repr__(self):
        return f'Vector({self.items!r})'


class DottedList:
    __slots__ = ('items', 'tail')

    def __init__(self, items, tail):
        self.items = items
        self.tail = tail

    def __eq__(self, other):
        return (isinstance(other, DottedList) and self.items == other.items
                and self.tail == other.tail)

    def __repr__(self):
        return f'DottedList({self.items!r}, {self.tail!r})'


_NAMED_CHARS_TO_INT = {
    'nul': 0, 'null': 0,
    'backspace': 8,
    'tab': 9,
    'newline': 10, 'linefeed': 10,
    'page': 12,
    'return': 13,
    'space': 32,
    'rubout': 127, 'delete': 127,
}
_INT_TO_NAMED_CHAR = {
    0: 'nul', 8: 'backspace', 9: 'tab', 10: 'newline', 12: 'page',
    13: 'return', 32: 'space', 127: 'rubout',
}

_DELIMS = set('()"; \t\n\r')


class _Reader:
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.len = len(text)

    def peek(self):
        return self.text[self.pos] if self.pos < self.len else ''

    def skip_ws(self):
        while self.pos < self.len and self.text[self.pos] in ' \t\n\r':
            self.pos += 1

    def read_datum(self):
        self.skip_ws()
        c = self.peek()
        if c == '':
            raise ValueError('unexpected end of input')
        if c == '(':
            self.pos += 1
            return self._read_list_body()
        if c == '"':
            return self._read_string()
        if c == '#':
            return self._read_hash()
        return self._read_atom()

    def _read_list_body(self):
        items = []
        tail = None
        while True:
            self.skip_ws()
            c = self.peek()
            if c == '':
                raise ValueError('unterminated list')
            if c == ')':
                self.pos += 1
                break
            if c == '.' and self._dot_is_delimited():
                self.pos += 1
                self.skip_ws()
                tail = self.read_datum()
                self.skip_ws()
                if self.peek() != ')':
                    raise ValueError('malformed dotted list')
                self.pos += 1
                break
            items.append(self.read_datum())
        if tail is None:
            return items
        return DottedList(items, tail)

    def _dot_is_delimited(self):
        nxt = self.text[self.pos + 1] if self.pos + 1 < self.len else ''
        return nxt == '' or nxt in _DELIMS

    def _read_string(self):
        assert self.peek() == '"'
        self.pos += 1
        out = []
        while True:
            c = self.peek()
            if c == '':
                raise ValueError('unterminated string')
            self.pos += 1
            if c == '"':
                break
            if c == '\\':
                esc = self.peek()
                self.pos += 1
                out.append({'n': '\n', 't': '\t', 'r': '\r', '"': '"',
                            '\\': '\\'}.get(esc, esc))
            else:
                out.append(c)
        return ''.join(out)

    def _read_hash(self):
        assert self.peek() == '#'
        nxt = self.text[self.pos + 1] if self.pos + 1 < self.len else ''
        if nxt == '(':
            self.pos += 2
            body = self._read_list_body()
            if isinstance(body, DottedList):
                raise ValueError('dotted vector is not valid')
            return Vector(body)
        if nxt == '\\':
            return self._read_char()
        if nxt in ('t', 'T', 'f', 'F'):
            token = self._read_token()
            if token in ('#t', '#true', '#T', '#True'):
                return True
            if token in ('#f', '#false', '#F', '#False'):
                return False
            raise ValueError(f'unrecognized boolean literal {token!r}')
        # `#%`-prefixed symbols (#%app, #%module-begin, ...) and anything
        # else starting with '#' fall back to plain symbol tokenizing.
        return self._read_atom()

    def _read_char(self):
        assert self.text[self.pos:self.pos + 2] == '#\\'
        self.pos += 2
        start = self.pos
        # A char literal is either exactly one character, or a run of
        # alphanumerics naming a char (e.g. "space", "newline").
        if self.pos < self.len and (self.text[self.pos].isalnum()):
            self.pos += 1
            while self.pos < self.len and self.text[self.pos].isalnum():
                self.pos += 1
        elif self.pos < self.len:
            self.pos += 1
        name = self.text[start:self.pos]
        if len(name) == 1:
            return Char(name)
        lname = name.lower()
        if lname in _NAMED_CHARS_TO_INT:
            return Char(chr(_NAMED_CHARS_TO_INT[lname]))
        if lname.startswith('u') and len(lname) > 1:
            try:
                return Char(chr(int(lname[1:], 16)))
            except ValueError:
                pass
        raise ValueError(f'unrecognized char literal #\\{name}')

    def _read_token(self):
        start = self.pos
        while self.pos < self.len and self.text[self.pos] not in _DELIMS:
            self.pos += 1
        return self.text[start:self.pos]

    def _read_atom(self):
        token = self._read_token()
        if token == '':
            raise ValueError('empty token')
        try:
            return int(token)
        except ValueError:
            pass
        try:
            return float(token)
        except ValueError:
            pass
        return Symbol(token)


def parse(text):
    """Parses exactly one datum from `text`, ignoring trailing whitespace."""
    reader = _Reader(text)
    datum = reader.read_datum()
    reader.skip_ws()
    if reader.pos != reader.len:
        raise ValueError(f'trailing input after datum: {text[reader.pos:]!r}')
    return datum


def _write_string(s):
    out = ['"']
    for c in s:
        if c == '"':
            out.append('\\"')
        elif c == '\\':
            out.append('\\\\')
        elif c == '\n':
            out.append('\\n')
        elif c == '\t':
            out.append('\\t')
        elif c == '\r':
            out.append('\\r')
        else:
            out.append(c)
    out.append('"')
    return ''.join(out)


def _write_char(ch):
    code = ord(ch.ch)
    if code in _INT_TO_NAMED_CHAR:
        return '#\\' + _INT_TO_NAMED_CHAR[code]
    if ch.ch.isprintable():
        return '#\\' + ch.ch
    return '#\\u%04X' % code


def write(d):
    """Renders `d` back to Racket `write`-compatible text."""
    if d is True:
        return '#t'
    if d is False:
        return '#f'
    if isinstance(d, int):
        return str(d)
    if isinstance(d, float):
        return repr(d)
    if isinstance(d, str):
        return _write_string(d)
    if isinstance(d, Symbol):
        return d.name
    if isinstance(d, Char):
        return _write_char(d)
    if isinstance(d, Vector):
        return '#(' + ' '.join(write(x) for x in d.items) + ')'
    if isinstance(d, DottedList):
        return ('(' + ' '.join(write(x) for x in d.items) + ' . '
                + write(d.tail) + ')')
    if isinstance(d, list):
        return '(' + ' '.join(write(x) for x in d) + ')'
    raise TypeError(f'cannot write datum of type {type(d)!r}: {d!r}')
