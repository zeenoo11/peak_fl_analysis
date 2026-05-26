"""Unwrap hard-wrapped prose in a markdown file.

Preserves code fences, tables, headers, horizontal rules, list-item structure,
and blockquotes (including nested code/lists/prose inside `> ` blocks).

Usage:
    uv run python tools/unwrap_md.py path/to/file.md [more.md ...]

Idempotent: running twice produces the same output as running once.
"""
import re
import sys
from pathlib import Path

CODE_FENCE = re.compile(r'^(\s*(?:>\s?)*)```')
LIST_ITEM = re.compile(r'^(\s*(?:>\s?)*)([-*+]|\d+\.)(\s+)(.*)$')
BLOCKQUOTE = re.compile(r'^(\s*(?:>\s?)+)(.*)$')


def strip_bq(line: str):
    m = BLOCKQUOTE.match(line)
    if m:
        return m.group(1), m.group(2)
    return '', line


def is_atomic(line: str) -> bool:
    """Lines that must not be merged with adjacent prose."""
    if not line.strip():
        return True
    _, rest = strip_bq(line)
    if not rest.strip():           # bare `>` line inside blockquote
        return True
    rest_l = rest.lstrip()
    if rest_l.startswith('#'):     # header
        return True
    if rest_l.startswith('|'):     # table row
        return True
    if re.match(r'^-{3,}\s*$', rest_l):  # horizontal rule / table sep
        return True
    return False


def fence(line: str) -> bool:
    return CODE_FENCE.match(line) is not None


def unwrap_text(text: str) -> str:
    lines = text.split('\n')
    n = len(lines)

    # Precompute: is this line INSIDE a fenced code block (excluding the fence lines)?
    in_fence = [False] * n
    state = False
    for i, ln in enumerate(lines):
        if fence(ln):
            in_fence[i] = state    # fence line itself is a boundary, not "inside"
            state = not state
        else:
            in_fence[i] = state

    out = []
    i = 0
    while i < n:
        ln = lines[i]
        if in_fence[i] or fence(ln) or is_atomic(ln):
            out.append(ln)
            i += 1
            continue

        # Collect paragraph: run of non-atomic, non-fence, non-in-fence lines.
        para = []
        while (i < n
               and not is_atomic(lines[i])
               and not fence(lines[i])
               and not in_fence[i]):
            para.append(lines[i])
            i += 1
        out.extend(_unwrap_para(para))

    return '\n'.join(out)


def _unwrap_para(para):
    """Process one paragraph: handle blockquote recursion + list items."""
    if all(BLOCKQUOTE.match(ln) for ln in para):
        # All lines share blockquote prefix → strip, recurse, re-prefix.
        prefix_first, _ = strip_bq(para[0])
        prefix = prefix_first.rstrip() + ' '
        stripped = [strip_bq(ln)[1] for ln in para]
        inner = _unwrap_para(stripped)
        return [
            (prefix + s) if s.strip() else prefix.rstrip()
            for s in inner
        ]

    # No (uniform) blockquote here — group list items and plain prose.
    items = []  # (prefix, [content_pieces])
    cur_prefix = None
    cur_pieces = []
    for ln in para:
        lm = LIST_ITEM.match(ln)
        if lm:
            if cur_prefix is not None:
                items.append((cur_prefix, cur_pieces))
            cur_prefix = lm.group(1) + lm.group(2) + lm.group(3)
            cur_pieces = [lm.group(4).rstrip()]
        else:
            content = ln.strip()
            if cur_prefix is None:
                cur_prefix = ''
                cur_pieces = [content]
            else:
                cur_pieces.append(content)
    if cur_prefix is not None:
        items.append((cur_prefix, cur_pieces))

    return [prefix + ' '.join(p for p in pieces if p)
            for prefix, pieces in items]


def main(argv):
    if not argv:
        print(__doc__)
        sys.exit(2)
    for p in argv:
        path = Path(p)
        original = path.read_text(encoding='utf-8')
        new = unwrap_text(original)
        if new != original:
            path.write_text(new, encoding='utf-8')
            print(f'unwrapped: {path}')
        else:
            print(f'no change: {path}')


if __name__ == '__main__':
    main(sys.argv[1:])
