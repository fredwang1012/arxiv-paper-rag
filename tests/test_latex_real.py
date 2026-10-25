"""Cases found by parsing real arXiv papers: bundled templates, custom macros, front matter."""
import time

from app import latex
from app.latex import latex_sections
from tests.test_latex import doc, make_tar


def test_main_file_is_the_one_that_pulls_in_the_most_text(tmp_path):
    template = doc("\\section{Submission Guidelines}\n" + "template words " * 100)
    paper = doc("\\section{Real}\\input{intro}\\input{more}")
    files = {"example_paper.tex": template, "paper.tex": paper, "intro.tex": "a " * 600, "more.tex": "b " * 600}
    s = latex_sections(make_tar(tmp_path, files))
    assert [p for p, _ in s] == [["1 Real"]]


def test_simple_macros_are_expanded_in_titles_and_text(tmp_path):
    pre = "\\newcommand{\\algoName}{NTP\\xspace}\n\\newcommand\\full{Neural Task Programming}\n\\newcommand{\\arg}[1]{#1}\n"
    body = "\\section{\\full}\nOur \\algoName model and \\algoName{} again."
    s = latex_sections(make_tar(tmp_path, {"a.tex": doc(body, pre)}))
    assert s == [(["1 Neural Task Programming"], "Our NTP model and NTP again.")]


def test_noindent_before_bold_run_in_heading(tmp_path):
    body = "\\section{S}\n\\noindent\\textbf{Setup.} We start."
    assert latex_sections(make_tar(tmp_path, {"a.tex": doc(body)})) == [(["1 S"], "Setup. We start.")]


def test_front_matter_macros_before_first_section_are_dropped(tmp_path):
    body = "\\title{A Paper}\n\\IEEEpeerreviewmaketitle\n\\section{Intro}\ntext"
    s = latex_sections(make_tar(tmp_path, {"a.tex": doc(body)}))
    assert [p for p, _ in s] == [["1 Intro"]]


# ---------- hostile sources: every limit is checked with a size or time bound ----------

def total_chars(sections) -> int:
    return sum(len(t) for _, t in sections)


def test_macro_expansion_cannot_amplify(tmp_path, monkeypatch):
    monkeypatch.setattr(latex, "MAX_BODY_CHARS", 200_000)
    pre = ("\\newcommand{\\a}{" + "x" * 150 + "}\n\\newcommand{\\b}{" + "\\a" * 90 + "}\n"
           "\\newcommand{\\c}{" + "\\b" * 90 + "}\n")
    body = "\\section{S}\n" + "\\c " * 30000  # 100 KB of uses of a macro that grows 150 * 90 * 90 times
    start = time.monotonic()
    s = latex_sections(make_tar(tmp_path, {"a.tex": doc(body, pre)}))
    assert time.monotonic() - start < 10
    assert total_chars(s) < 400_000


def test_long_macro_bodies_are_not_treated_as_simple(tmp_path):
    pre = "\\newcommand{\\big}{" + "y" * 300 + "}\n"
    s = latex_sections(make_tar(tmp_path, {"a.tex": doc("\\section{S}\nsee \\big here", pre)}))
    assert s == [(["1 S"], "see \\big here")]


def test_unclosed_macro_definitions_scan_in_bounded_time(tmp_path):
    pre = "\\newcommand{\\m}{" * 20000 + "".join(f"\\def\\n{i}{{ " for i in range(20000))
    start = time.monotonic()
    latex_sections(make_tar(tmp_path, {"a.tex": doc("\\section{S}\nx", pre)}))
    assert time.monotonic() - start < 5


def test_many_main_candidates_and_includes_are_bounded(tmp_path):
    files = {f"f{i}.tex": doc("\\section{S}\n" + "\\input{missing}" * 200) for i in range(300)}
    start = time.monotonic()
    s = latex_sections(make_tar(tmp_path, files))
    assert time.monotonic() - start < 10
    assert s == []  # the sections are empty: includes were all missing


def test_every_input_match_is_charged_to_the_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(latex, "MAX_INLINE_OPS", 50)
    lookups = []
    real = latex._lookup
    monkeypatch.setattr(latex, "_lookup", lambda *a: lookups.append(1) or real(*a))
    latex_sections(make_tar(tmp_path, {"a.tex": doc("\\section{S}\n" + "\\input{nothing}" * 5000)}))
    assert len(lookups) <= 50


def test_skipped_members_count_toward_the_decompression_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(latex, "MAX_TOTAL_BYTES", 100_000)
    monkeypatch.setattr(latex, "TAR_SLACK_BYTES", 10_000)
    files = {"main.tex": doc("\\section{S}\nkept"), "big.bin": b"\0" * 5_000_000, "late.tex": doc("\\section{L}\nz")}
    folder = make_tar(tmp_path, files)
    got = latex._read_tex_files(folder)
    assert set(got) == {"main.tex"}  # stopped at the bomb; what was read before it is kept
    assert latex_sections(folder) == [(["1 S"], "kept")]


def test_bomb_before_the_tex_files_gives_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(latex, "MAX_TOTAL_BYTES", 100_000)
    monkeypatch.setattr(latex, "TAR_SLACK_BYTES", 10_000)
    folder = make_tar(tmp_path, {"big.bin": b"\0" * 5_000_000, "main.tex": doc("\\section{S}\nx")})
    assert latex._read_tex_files(folder) == {}
