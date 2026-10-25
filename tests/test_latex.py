"""LaTeX parsing: main file choice, \\input inlining, comments, numbering, text cleanup."""
import gzip
import io
import tarfile
import time

from app import latex
from app.latex import latex_sections


def make_tar(folder, files: dict[str, str | bytes]):
    folder.mkdir(parents=True, exist_ok=True)
    with tarfile.open(folder / "source.tar.gz", "w:gz") as tar:
        for name, content in files.items():
            data = content.encode("utf-8") if isinstance(content, str) else content
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return folder


def doc(body: str, pre: str = "") -> str:
    return "\\documentclass{article}\n" + pre + "\\begin{document}\n" + body + "\n\\end{document}\n"


def by_path(sections):
    return {" > ".join(p): t for p, t in sections}


def test_numbering_starred_and_subsection_before_section(tmp_path):
    body = r"""
\subsection{Orphan}
o text
\section{Model}
m text
\subsection{Inference}
i text
\subsubsection{Detail}
d text
\subsection*{Unnumbered Sub}
u text
\section*{Notes}
n text
\section{Data}
x text
\subsection{Again}
a text
"""
    s = latex_sections(make_tar(tmp_path, {"a.tex": doc(body)}))
    assert [p for p, _ in s] == [
        ["0.1 Orphan"],
        ["1 Model"],
        ["1 Model", "1.1 Inference"],
        ["1 Model", "1.1 Inference", "1.1.1 Detail"],
        ["1 Model", "Unnumbered Sub"],
        ["Notes"],
        ["2 Data"],
        ["2 Data", "2.1 Again"],
    ]


def test_appendix_letters(tmp_path):
    body = "\\section{Intro}\ni\n\\appendix\n\\section{Proofs}\np\n\\subsection{Lemma}\nl\n\\section{More}\nm"
    s = latex_sections(make_tar(tmp_path, {"a.tex": doc(body)}))
    assert [p for p, _ in s] == [["1 Intro"], ["A Proofs"], ["A Proofs", "A.1 Lemma"], ["B More"]]


def test_abstract_and_introduction_preface(tmp_path):
    body = "\\maketitle\n\\begin{abstract}\nWe study \\emph{things}.\n\\end{abstract}\nPreface words.\n\\section{Next}\nbody"
    s = latex_sections(make_tar(tmp_path, {"a.tex": doc(body)}))
    assert s[0] == (["Abstract"], "We study things.")
    assert s[1] == (["Introduction"], "Preface words.")
    assert s[2][0] == ["1 Next"]


def test_empty_preface_is_dropped(tmp_path):
    s = latex_sections(make_tar(tmp_path, {"a.tex": doc("\\maketitle\n\\section{Only}\nt")}))
    assert [p for p, _ in s] == [["1 Only"]]


def test_main_file_prefers_main_tex_then_ms_then_largest(tmp_path):
    small = doc("\\section{Small}\ns")
    big = doc("\\section{Big}\n" + "b " * 200)
    s = latex_sections(make_tar(tmp_path / "1", {"a.tex": big, "main.tex": small}))
    assert s[0][0] == ["1 Small"]
    s = latex_sections(make_tar(tmp_path / "2", {"a.tex": big, "ms.tex": small}))
    assert s[0][0] == ["1 Small"]
    s = latex_sections(make_tar(tmp_path / "3", {"a.tex": small, "b.tex": big, "frag.tex": "\\section{Frag}\nf"}))
    assert s[0][0] == ["1 Big"]


def test_no_documentclass_means_no_sections(tmp_path):
    assert latex_sections(make_tar(tmp_path, {"a.tex": "\\section{X}\nx"})) == []


def test_input_inlining_nested_missing_and_cycle(tmp_path):
    files = {
        "main.tex": doc("\\input{intro}\n\\include{sec/model.tex}\n\\input{missing}\n\\input{loop}"),
        "intro.tex": "\\section{Intro}\nhello \\input{extra}",
        "extra.tex": " world",
        "sec/model.tex": "\\section{Model}\nmodel text",
        "loop.tex": "\\section{Loop}\nloop \\input{loop}\\input{main}",
    }
    s = by_path(latex_sections(make_tar(tmp_path, files)))
    assert s["1 Intro"] == "hello world"
    assert s["2 Model"] == "model text"
    assert s["3 Loop"] == "loop"


def test_input_depth_is_capped(tmp_path):
    files = {"main.tex": doc("\\section{S}\n\\input{f0}")}
    for i in range(20):
        files[f"f{i}.tex"] = f"L{i} \\input{{f{i + 1}}}"
    text = by_path(latex_sections(make_tar(tmp_path, files)))["1 S"]
    assert "L0" in text and "L9" in text and "L15" not in text


def test_comments_stripped_but_escaped_percent_kept(tmp_path):
    body = "\\section{A}\nGain of 5\\% here. % hidden note\n% \\section{Ghost}\nline two\\\\% after break"
    s = latex_sections(make_tar(tmp_path, {"a.tex": doc(body)}))
    assert [p for p, _ in s] == [["1 A"]]
    assert s[0][1] == "Gain of 5% here. line two"


def test_cite_ref_label_formatting_and_math(tmp_path):
    body = (
        "\\section{Model}\\label{sec:model}\n"
        "As in \\citet{reed2015} and \\cite[p.~3]{a,b}, see Sec.~\\ref{sec:x} and \\textbf{very \\emph{bold}} ideas.\n"
        "Inline $x^2 + y$ and display\n\\begin{equation}\\label{eq:1} E = mc^2 \\end{equation}\n"
        "and \\[ a = b \\] done.\n"
    )
    t = latex_sections(make_tar(tmp_path, {"a.tex": doc(body)}))[0][1]
    assert t == ("As in [cite] and [cite], see Sec. ? and very bold ideas. "
                 "Inline $x^2 + y$ and display $E = mc^2$ and $a = b$ done.")


def test_figures_and_tables_keep_only_captions(tmp_path):
    body = (
        "\\section{Results}\nBefore.\n"
        "\\begin{figure}[t]\\centering\\includegraphics{x.png}\\caption{Overview of \\textit{TACO}.}"
        "\\label{fig:o}\\end{figure}\n"
        "\\begin{table*}\\begin{tabular}{ll}a & b\\\\\\end{tabular}\\caption{Scores}\\end{table*}\n"
        "After."
    )
    t = latex_sections(make_tar(tmp_path, {"a.tex": doc(body)}))[0][1]
    assert t == "Before. Overview of TACO. Scores After."


def test_preamble_and_bibliography_removed(tmp_path):
    text = doc("\\section{A}\nbody\n\\bibliographystyle{plain}\n\\bibliography{refs}\n\\section{After}\nz",
               pre="\\title{Secret Title}\n\\newcommand{\\foo}{bar}\n")
    s = latex_sections(make_tar(tmp_path, {"a.tex": text}))
    assert s == [(["1 A"], "body"), (["2 After"], "z")]  # only \end{document} ends the body


def test_thebibliography_removed(tmp_path):
    body = "\\section{A}\nbody\n\\begin{thebibliography}{9}\n\\bibitem{x} Someone.\n\\end{thebibliography}"
    assert latex_sections(make_tar(tmp_path, {"a.tex": doc(body)})) == [(["1 A"], "body")]


def test_single_gzipped_tex(tmp_path):
    (tmp_path / "source.gz").write_bytes(gzip.compress(doc("\\section{One}\nsolo").encode()))
    assert latex_sections(tmp_path) == [(["1 One"], "solo")]


def test_single_gzipped_non_latex_is_not_latex(tmp_path):
    (tmp_path / "source.gz").write_bytes(gzip.compress(b"just some text, no class"))
    assert latex_sections(tmp_path) == []


def test_source_none_and_missing_source(tmp_path):
    assert latex_sections(tmp_path) == []
    (tmp_path / "source.none").write_bytes(b"")
    assert latex_sections(tmp_path) == []


def test_corrupt_archive_gives_no_sections(tmp_path):
    (tmp_path / "source.tar.gz").write_bytes(b"not a gzip at all")
    assert latex_sections(tmp_path) == []


def test_appendix_after_bibliography_is_kept(tmp_path):
    body = ("\\section{A}x\\bibliographystyle{plain}\\bibliography{r}\n"
            "\\begin{thebibliography}{9}\\bibitem{x} Someone.\\end{thebibliography}\n"
            "\\appendix\\section{P}proof")
    s = latex_sections(make_tar(tmp_path, {"a.tex": doc(body)}))
    assert s == [(["1 A"], "x"), (["A P"], "proof")]


def test_include_amplification_is_bounded(tmp_path):
    files = {"main.tex": doc("\\section{S}\n\\input{f0}")}
    for i in range(12):
        files[f"f{i}.tex"] = ("y" * 2000 + f"\\input{{f{i + 1}}}") * 10  # 10 children each: 10**12 expansions
    files["f12.tex"] = "end"
    start = time.monotonic()
    s = latex_sections(make_tar(tmp_path, files))
    assert time.monotonic() - start < 20
    assert sum(len(t) for _, t in s) < 8_000_000


def test_oversized_gzip_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(latex, "MAX_TEX_BYTES", 1000)
    (tmp_path / "source.gz").write_bytes(gzip.compress(doc("\\section{S}\n" + "x" * 5000).encode()))
    assert latex_sections(tmp_path) == []


def test_oversized_tar_member_is_skipped_without_reading(tmp_path, monkeypatch):
    monkeypatch.setattr(latex, "MAX_TEX_BYTES", 1000)
    make_tar(tmp_path, {"big.tex": doc("\\section{Big}\n" + "x" * 5000), "main.tex": doc("\\section{Small}\nok")})
    reads = []
    real = tarfile.TarFile.extractfile
    monkeypatch.setattr(tarfile.TarFile, "extractfile", lambda self, m: reads.append(m.name) or real(self, m))
    assert latex_sections(tmp_path) == [(["1 Small"], "ok")]
    assert reads == ["main.tex"]


def test_too_many_members_stops_reading(tmp_path, monkeypatch):
    monkeypatch.setattr(latex, "MAX_MEMBERS", 3)
    files = {f"a{i}.txt": "junk" for i in range(3)} | {"main.tex": doc("\\section{S}\nx")}
    assert latex_sections(make_tar(tmp_path, files)) == []


def test_unclosed_environments_and_deep_nesting_are_fast(tmp_path):
    body = ("\\section{S}\n" + "\\begin{figure} a \\caption{c " * 5000 + "\\[ x " * 5000
            + "\\textbf{" * 20000 + "\\begin{equation} " * 5000)
    start = time.monotonic()
    latex_sections(make_tar(tmp_path, {"a.tex": doc(body)}))
    assert time.monotonic() - start < 1


def test_latin1_source_is_decoded(tmp_path):
    body = "\\section{Caf\xe9}\n" + "na\xefve r\xe9sum\xe9 " * 20
    s = latex_sections(make_tar(tmp_path, {"a.tex": doc(body).encode("latin-1")}))
    assert s[0][0] == ["1 Caf\xe9"]
    assert s[0][1].startswith("na\xefve r\xe9sum\xe9")


def _parses_fast(tmp_path, body):
    start = time.monotonic()
    latex_sections(make_tar(tmp_path, {"a.tex": doc(body)}))
    assert time.monotonic() - start < 1


def test_unclosed_abstract_repeated_is_fast(tmp_path):
    _parses_fast(tmp_path, "\\section{S}\n" + "\\begin{abstract} words " * 10000)


def test_unclosed_section_option_repeated_is_fast(tmp_path):
    _parses_fast(tmp_path, "\\section[ " * 20000)


def test_unclosed_cite_option_repeated_is_fast(tmp_path):
    _parses_fast(tmp_path, "\\section{S}\n" + "\\cite[ " * 30000)


def test_abstract_ends_at_first_closer_after_opener(tmp_path):
    s = by_path(latex_sections(make_tar(tmp_path, {"a.tex": doc(
        "\\end{abstract} stray \\begin{abstract} real text \\end{abstract}\n\\section{S}\nbody")})))
    assert s["Abstract"] == "real text"
