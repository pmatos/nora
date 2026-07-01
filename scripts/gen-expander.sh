#!/usr/bin/env bash
#
# gen-expander.sh — regenerate expander/expander.rktl from Racket source.
#
# expander/expander.rktl is Racket's expander demodularized: every module of
# the expander fully expanded and flattened into a single (linklet ...) form.
# NORA lexes/parses/interprets it, so it must track the Racket release it was
# produced from.
#
# This script is self-contained: given a host `racket` (plus git and a C
# compiler) it shallow-clones the matching Racket source tag, builds `zuo`, and
# runs the expander's own `expander-src` build target using the host racket as
# the flattening engine — no full Chez/Racket build required. The result is
# copied over expander/expander.rktl.
#
# By default it pins the source to the tag matching the installed racket
# (e.g. v9.2). Override with an explicit ref:
#
#   scripts/gen-expander.sh            # match installed racket
#   scripts/gen-expander.sh v9.2       # force a specific tag/branch/commit
#
# Environment overrides:
#   RACKET  racket executable to use as host   (default: racket on PATH)
#   CC      C compiler for building zuo         (default: cc)
#   KEEP_BUILD=1  keep the temporary build tree (default: removed on exit)

set -euo pipefail

log() { printf '>> %s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$REPO_ROOT/expander/expander.rktl"

RACKET="${RACKET:-racket}"
CC="${CC:-cc}"

command -v git >/dev/null 2>&1 || die "git not found on PATH"
command -v "$CC" >/dev/null 2>&1 || die "C compiler '$CC' not found (set \$CC)"
command -v "$RACKET" >/dev/null 2>&1 || die "racket '$RACKET' not found (set \$RACKET)"

# raco is derived from the host racket by the expander build; it needs
# compiler-lib (compiler/cm) to compile the expander sources.
"$RACKET" -l compiler/cm >/dev/null 2>&1 \
  || die "host racket lacks compiler/cm (install the 'compiler-lib' package)"

# Version string reported by the host, e.g. "Welcome to Racket v9.2 [cs]."
HOST_VERSION="$("$RACKET" --version | grep -oE 'v[0-9]+(\.[0-9]+)+' | head -1)"
[ -n "$HOST_VERSION" ] || die "could not parse version from: $("$RACKET" --version)"

# Default ref = tag matching the host's major.minor (release tags are two-part,
# e.g. v9.2). An explicit argument overrides this entirely.
DEFAULT_REF="$(printf '%s' "$HOST_VERSION" | grep -oE 'v[0-9]+\.[0-9]+' | head -1)"
REF="${1:-$DEFAULT_REF}"

log "host racket:  $("$RACKET" --version)"
log "source ref:   $REF"
if [ "$REF" != "$DEFAULT_REF" ]; then
  log "WARNING: source ref '$REF' differs from host racket ($DEFAULT_REF);"
  log "         the flattened expander may not match your runtime."
fi

BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/nora-gen-expander.XXXXXX")"
cleanup() { [ "${KEEP_BUILD:-0}" = 1 ] || rm -rf "$BUILD_DIR"; }
trap cleanup EXIT
[ "${KEEP_BUILD:-0}" = 1 ] && log "build tree kept at: $BUILD_DIR"

SRC="$BUILD_DIR/racket"
log "cloning racket/$REF (shallow) ..."
git clone --quiet --depth 1 --branch "$REF" \
  https://github.com/racket/racket "$SRC" \
  || die "clone failed — is '$REF' a valid tag/branch? (try passing an explicit ref)"

ZUO_SRC="$SRC/racket/src/zuo"
EXPANDER_DIR="$SRC/racket/src/expander"
[ -f "$ZUO_SRC/zuo.c" ] || die "zuo.c not found in checkout ($ZUO_SRC)"
[ -f "$EXPANDER_DIR/main.zuo" ] || die "expander build not found ($EXPANDER_DIR)"

log "building zuo ..."
"$CC" -O2 -DZUO_LIB_PATH="\"$ZUO_SRC/lib\"" -o "$BUILD_DIR/zuo" "$ZUO_SRC/zuo.c"

log "flattening expander (this runs the host racket) ..."
( cd "$EXPANDER_DIR" && "$BUILD_DIR/zuo" main.zuo expander-src racket="$(command -v "$RACKET")" )

GENERATED="$EXPANDER_DIR/compiled/expander.rktl"
[ -f "$GENERATED" ] || die "build finished but $GENERATED is missing"

# Cheap structural check: it must read as a single (linklet ...) datum.
"$RACKET" -e "(call-with-input-file \"$GENERATED\"
                (lambda (p)
                  (define d (read p))
                  (unless (and (pair? d) (eq? (car d) 'linklet) (eof-object? (read p)))
                    (error 'gen-expander \"not a single linklet form\"))))" \
  || die "generated file failed the linklet sanity check"

mkdir -p "$(dirname "$OUT")"
cp "$GENERATED" "$OUT"
log "wrote $OUT ($(wc -c <"$OUT") bytes) from racket $REF"
