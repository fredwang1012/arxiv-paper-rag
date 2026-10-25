"""Turn a paper's LaTeX source (gzipped tar or single gzipped .tex) into (section path, plain text) pairs."""
import gzip
import logging
import re
import tarfile
import zlib
from pathlib import Path, PurePosixPath

log = logging.getLogger(__name__)

MAX_INPUT_DEPTH = 10
# Limits against hostile or broken archives: decompression bombs and \input amplification
MAX_TEX_BYTES = 20_000_000  # one .tex member, or the decompressed single file
MAX_TOTAL_BYTES = 50_000_000
TAR_SLACK_BYTES = 1_000_000  # tar headers and padding, on top of the member sizes
MAX_MEMBERS = 5_000
MAX_MAIN_CANDIDATES = 20
MAX_INLINE_CHARS = 5_000_000  # total size of files pulled in by \input/\include
MAX_INLINE_OPS = 2_000
MAX_BODY_CHARS = 8_000_000
MAX_SECTION_CHARS = 2_000_000
MAX_UNWRAP_PASSES = 20
_LEVELS = {"section": 0, "subsection": 1, "subsubsection": 2}
_PREFERRED_MAIN = ("main.tex", "ms.tex")


# ---------- reading the source ----------

def _decode(data: bytes) -> str:
    text = data.decode("utf-8", errors="replace")
    if text.count("\ufffd") > max(5, len(text) // 200):  # many bad bytes: probably latin-1, not utf-8
        text = data.decode("latin-1")
    return text


def _strip_comments(text: str) -> str:
    # an odd number of backslashes before % escapes it; an even number (e.g. a "\\" line break) does not
    return re.sub(r"(?<!\\)((?:\\\\)*)%[^\n]*", r"\1", text)


def _norm(name: str) -> str:
    return PurePosixPath(name.replace("\\", "/")).as_posix().removeprefix("./").lstrip("/")


class _TooBig(Exception):
    pass


class _LimitedReader:
    """File-like wrapper that raises once more than `limit` bytes have been read from it."""

    def __init__(self, fileobj, limit: int):
        self._f, self._left = fileobj, limit

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._left + 1
        data = self._f.read(min(size, self._left + 1))
        self._left -= len(data)
        if self._left < 0:
            raise _TooBig
        return data


def _read_tex_files(folder: Path) -> dict[str, str]:
    """name -> comment-free text of every .tex file in the source; empty when there is no usable source."""
    tar_path, gz_path = folder / "source.tar.gz", folder / "source.gz"  # not source.* : source.none must not match
    try:
        if tar_path.exists():
            files, total = {}, 0
            try:
                # streaming mode over a byte-limited reader: members are read in memory, never extracted, and
                # skipped members still count because they are decompressed to reach the next header
                with gzip.open(tar_path, "rb") as gz, \
                        tarfile.open(fileobj=_LimitedReader(gz, MAX_TOTAL_BYTES + TAR_SLACK_BYTES), mode="r|") as tar:
                    for count, member in enumerate(tar):
                        total += max(member.size, 0)
                        if count >= MAX_MEMBERS or total > MAX_TOTAL_BYTES:
                            log.warning("LaTeX source in %s is too large; reading only part of it", folder)
                            break
                        if not (member.isfile() and member.name.lower().endswith(".tex")):
                            continue
                        if member.size > MAX_TEX_BYTES:
                            log.warning("skipping oversized %s in %s", member.name, folder)
                            continue
                        files[_norm(member.name)] = _decode(tar.extractfile(member).read())
            except _TooBig:
                log.warning("LaTeX source in %s decompresses to too much data; using what was read", folder)
        elif gz_path.exists():
            data = zlib.decompressobj(wbits=31).decompress(gz_path.read_bytes(), MAX_TEX_BYTES + 1)
            if len(data) > MAX_TEX_BYTES:
                log.warning("source.gz in %s is too large; ignoring it", folder)
                return {}
            files = {"main.tex": _decode(data)}
        else:
            return {}
    except (OSError, EOFError, tarfile.TarError, zlib.error, MemoryError) as e:
        log.warning("could not read LaTeX source in %s: %s", folder, e)
        return {}
    return {name: _strip_comments(text) for name, text in files.items()}


def _basenames(files: dict[str, str]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for n in files:
        base = PurePosixPath(n).name
        index.setdefault(base, []).append(n)
        if base.endswith(".tex"):
            index.setdefault(base[:-4], []).append(n)
    return index


def _lookup(files: dict[str, str], by_base: dict[str, list[str]], target: str) -> str | None:
    target = _norm(target)
    for name in (target, target + ".tex"):
        if name in files:
            return name
    matches = by_base.get(PurePosixPath(target).name, [])
    return matches[0] if len(matches) == 1 else None


def _new_budget() -> dict:
    return {"chars": 0, "ops": 0, "warned": False}


def _inline(files: dict[str, str], main: str, by_base: dict[str, list[str]], budget: dict | None = None) -> str:
    """Expand \\input/\\include recursively; a shared budget stops repeated includes from multiplying the text."""
    budget = budget if budget is not None else _new_budget()

    def expand(name: str, stack: tuple[str, ...]) -> str:
        def repl(m: re.Match) -> str:
            if budget["ops"] >= MAX_INLINE_OPS or budget["chars"] >= MAX_INLINE_CHARS:
                if not budget["warned"]:
                    budget["warned"] = True
                    log.warning("\\input expansion budget exhausted; dropping further includes")
                return ""
            budget["ops"] += 1  # every match is charged, found or not
            found = _lookup(files, by_base, m.group(1).strip())
            if found is None or found in stack or len(stack) > MAX_INPUT_DEPTH:  # missing, cycle or too deep
                return ""
            budget["chars"] += len(files[found])
            return "\n" + expand(found, stack + (found,)) + "\n"

        return re.sub(r"\\(?:input|include)\s*\{([^{}]*)\}", repl, files[name])

    return expand(main, (main,))


def _find_main(files: dict[str, str]) -> tuple[str, str] | None:
    """The main file and its text with \\input/\\include expanded; None when no file has a \\documentclass."""
    candidates = [n for n, t in files.items() if r"\documentclass" in t]
    if not candidates:
        return None
    by_base = _basenames(files)
    for preferred in _PREFERRED_MAIN:
        for n in sorted(candidates):
            if PurePosixPath(n).name.lower() == preferred:
                return n, _inline(files, n, by_base)
    # the real paper is the one that pulls in the most text (a bundled style-guide example can be longer on its own);
    # only the largest few files are tried, sharing one inlining budget
    budget = _new_budget()
    best: tuple[str, str] | None = None
    for n in sorted(candidates, key=lambda n: len(files[n]), reverse=True)[:MAX_MAIN_CANDIDATES]:
        text = _inline(files, n, by_base, budget)
        if best is None or len(text) > len(best[1]):
            best = n, text
    return best


def _replace_pairs(text: str, opener: re.Pattern, closer, fn) -> str:
    """Replace each opener...closer span with fn(match, inner). Linear: when a closer is missing the opener is
    left alone, and later openers needing the same closer are not searched for again."""
    out, pos, missing = [], 0, set()
    for m in opener.finditer(text):
        close = closer(m)
        if m.start() < pos or close in missing:
            continue
        end = text.find(close, m.end())
        if end < 0:
            missing.add(close)
            continue
        out += [text[pos:m.start()], fn(m, text[m.end():end])]
        pos = end + len(close)
    out.append(text[pos:])
    return "".join(out)


_MACRO_DEF = re.compile(r"\\(?:newcommand|renewcommand|providecommand)\*?\s*\{?\\([A-Za-z]+)\}?\s*(?=[{\[])"
                        r"|\\def\s*\\([A-Za-z]+)\s*(?=\{)")
_MACRO_USE = re.compile(r"\\([A-Za-z]+)(\{\})?")
MAX_MACROS = 2_000  # definitions looked at, counting repeats
MAX_MACRO_BODY = 200
_MACRO_SCAN = 500  # give up looking for the closing brace of a definition after this many characters


def _short_body(text: str, pos: int) -> str | None:
    """Body of the {...} group at pos if it closes within MAX_MACRO_BODY characters, else None."""
    depth = 0
    for i in range(pos, min(len(text), pos + _MACRO_SCAN)):
        if text[i] == "{" and text[i - 1] != "\\":
            depth += 1
        elif text[i] == "}" and text[i - 1] != "\\":
            depth -= 1
            if depth == 0:
                return text[pos + 1:i] if i - pos - 1 <= MAX_MACRO_BODY else None
    return None


def _expand_macros(text: str) -> str:
    """Substitute the paper's own short argument-free macros, e.g. \\newcommand{\\model}{NPI}, so names and titles
    read well. Work and output are bounded: a hostile source cannot make the text grow past MAX_BODY_CHARS."""
    macros: dict[str, str] = {}
    for attempts, m in enumerate(_MACRO_DEF.finditer(text)):
        if attempts >= MAX_MACROS:
            break
        name = m.group(1) or m.group(2)
        if name in macros or not text.startswith("{", m.end()):  # "[" would mean the macro takes arguments
            continue
        body = _short_body(text, m.end())
        if body is not None:
            macros[name] = body
    if not macros:
        return text
    for _ in range(2):  # a macro body may use another macro
        out, pos, size = [], 0, len(text)
        for m in _MACRO_USE.finditer(text):
            body = macros.get(m.group(1))
            if body is None:
                continue
            size += len(body) - len(m.group(0))
            if size > MAX_BODY_CHARS:  # stop expanding; the remaining uses stay as written
                break
            out += [text[pos:m.start()], body]
            pos = m.end()
        out.append(text[pos:])
        text = "".join(out)
    return text


def _body(text: str) -> str:
    """Text between \\begin{document} and \\end{document} without the bibliography (appendices after it stay)."""
    start = re.search(r"\\begin\{document\}", text)
    if start:
        text = text[start.end():]
    end = text.find("\\end{document}")
    if end >= 0:
        text = text[:end]
    text = re.sub(r"\\bibliography(?:style)?\s*\{[^{}]*\}", "", text)
    text = _replace_pairs(text, re.compile(r"\\begin\{thebibliography\}"),
                          lambda m: "\\end{thebibliography}", lambda m, inner: "")
    return text[:MAX_BODY_CHARS]


# ---------- splitting into sections ----------

def _braced(text: str, pos: int) -> tuple[str, int]:
    """Contents of the {...} group whose "{" is at pos, and the index after its closing brace."""
    depth = 0
    for i in range(pos, len(text)):
        escaped = i > 0 and text[i - 1] == "\\"
        if text[i] == "{" and not escaped:
            depth += 1
        elif text[i] == "}" and not escaped:
            depth -= 1
            if depth == 0:
                return text[pos + 1:i], i + 1
    return text[pos + 1:], len(text)


_HEADING = re.compile(r"\\(section|subsection|subsubsection)(\*?)\s*(?:\[[^\]]*\])?\s*(?=\{)|\\appendix\b")


def _number(counters: list[int], level: int, appendix: bool) -> str:
    parts = [str(c) for c in counters[:level + 1]]
    if appendix and 1 <= counters[0] <= 26:
        parts[0] = chr(ord("A") + counters[0] - 1)
    return ".".join(parts)


def _split(body: str) -> list[tuple[list[str], str]]:
    """Cut the body at headings into (path, raw text); text before the first heading gets the path ["Introduction"]."""
    pieces: list[tuple[list[str], str]] = []
    path: list[tuple[int, str]] = []  # (level, "number title") for the headings in force
    counters = [0, 0, 0]
    appendix = False
    current = ["Introduction"]
    pos = 0

    for m in _HEADING.finditer(body):
        if m.start() < pos:  # inside a title already consumed
            continue
        pieces.append((current, body[pos:m.start()]))
        if m.group(0).startswith(r"\appendix"):
            appendix, counters = True, [0, 0, 0]
            pos = m.end()
            continue
        level = _LEVELS[m.group(1)]
        title_raw, pos = _braced(body, m.end())
        title = clean_text(title_raw)
        if m.group(2) == "":
            counters[level] += 1
            for deeper in range(level + 1, 3):
                counters[deeper] = 0
            element = f"{_number(counters, level, appendix)} {title}"
        else:
            element = title
        path = [p for p in path if p[0] < level] + [(level, element)]
        current = [p[1] for p in path]
    pieces.append((current, body[pos:]))
    return pieces


# ---------- LaTeX to text ----------

_MATH_ENVS = "equation|align|gather|multline|eqnarray|displaymath|math"
_MATH_OPEN = re.compile(rf"\\\[|\\\(|\\begin\{{({_MATH_ENVS}\*?)\}}")
_MATH_DOLLARS = re.compile(r"\$\$.+?\$\$|(?<!\\)\$.+?(?<!\\)\$", re.S)
_FLOAT_OPEN = re.compile(r"\\begin\{((?:figure|table|wrapfigure|wraptable|sidewaysfigure|sidewaystable)\*?)\}")
_FORMATTING = ("textbf|textit|emph|texttt|textsc|textrm|textsf|textsl|textup|underline|mbox|footnote|footnotetext"
               "|url|texorpdfstring|textnormal|uppercase|lowercase|centerline|paragraph|subparagraph")
_CITE = r"\\(?:cite[a-zA-Z]*|nocite|parencite|textcite)\*?\s*(?:\[[^\]]*\]\s*)*\{[^{}]*\}"
_REF = r"\\(?:ref|eqref|autoref|cref|Cref|pageref|nameref|vref)\*?\s*\{[^{}]*\}"


def _dollars(inner: str) -> str:
    inner = re.sub(r"\\label\{[^{}]*\}", "", inner)
    return "$" + " ".join(inner.split()) + "$"


def _math_closer(m: re.Match) -> str:
    return {"\\[": "\\]", "\\(": "\\)"}.get(m.group(0)) or "\\end{" + m.group(1) + "}"


def _captions(m: re.Match, inner: str) -> str:
    """Keep only the \\caption{} text of a figure or table environment."""
    out, pos = [], 0
    for c in re.finditer(r"\\caption\*?\s*(?:\[[^\]]*\])?\s*(?=\{)", inner):
        if c.start() < pos:
            continue
        cap, pos = _braced(inner, c.end())
        out.append(cap)
    return "\n\n" + "\n\n".join(out) + "\n\n" if out else "\n\n"


def clean_text(raw: str) -> str:
    """Light LaTeX-to-text: math kept as $...$, citations and refs replaced, formatting macros unwrapped."""
    text = _replace_pairs(raw, _FLOAT_OPEN, lambda m: "\\end{" + m.group(1) + "}", _captions)
    maths: list[str] = []

    def stash(inner: str) -> str:
        maths.append(_dollars(inner))
        return f"\x00{len(maths) - 1}\x00"

    # protect math from the rewrites below
    text = _replace_pairs(text, _MATH_OPEN, _math_closer, lambda m, inner: stash(inner))
    text = _MATH_DOLLARS.sub(lambda m: stash(m.group(0).strip("$") if m.group(0).startswith("$$") else m.group(0)[1:-1]),
                             text)
    text = re.sub(r"\\label\s*\{[^{}]*\}", "", text)
    text = re.sub(_CITE, "[cite]", text)
    text = re.sub(_REF, "?", text)
    text = re.sub(r"\\href\s*\{[^{}]*\}\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\(?:noindent|xspace|centering|maketitle|newpage|clearpage|bigskip|medskip|smallskip|par)\b\s*", " ", text)
    unwrap = re.compile(rf"\\(?:{_FORMATTING})\*?\s*\{{([^{{}}]*)\}}")
    for _ in range(MAX_UNWRAP_PASSES):  # innermost first, so nested macros unwrap
        new = unwrap.sub(r"\1", text)
        if new == text:
            break
        text = new
    text = re.sub(r"\\(?:vspace|hspace|vskip|hskip)\*?\s*\{[^{}]*\}", " ", text)
    text = re.sub(r"\\item\b\s*(?:\[[^\]]*\])?", "\n- ", text)
    text = re.sub(r"\\(?:begin|end)\s*\{[^{}]*\}(?:\[[^\]]*\])?", " ", text)
    text = text.replace("\\\\", " ").replace("~", " ").replace("``", '"').replace("''", '"')
    text = re.sub(r"\\([%&_#$])", r"\1", text)
    text = re.sub(r"\\([{}])", r"\1", text).replace("{", "").replace("}", "")
    text = re.sub(r"\x00(\d+)\x00", lambda m: maths[int(m.group(1))], text)
    return " ".join(text.split())


# ---------- entry point ----------

def latex_sections(folder: Path) -> list[tuple[list[str], str]]:
    """(path, text) for each non-empty section; empty when the paper has no usable LaTeX source."""
    files = _read_tex_files(folder)
    found = _find_main(files)
    if found is None:
        return []
    body = _body(_expand_macros(found[1]))

    sections: list[tuple[list[str], str]] = []
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S)
    if abstract:
        sections.append((["Abstract"], clean_text(abstract.group(1))))
        body = body[:abstract.start()] + body[abstract.end():]
    for i, (path, raw) in enumerate(_split(body)):
        text = clean_text(raw[:MAX_SECTION_CHARS])
        if i == 0 and "\\" in text:  # title/author blocks and publisher macros, not prose
            continue
        sections.append((path, text))
    return [(p, t) for p, t in sections if t]
