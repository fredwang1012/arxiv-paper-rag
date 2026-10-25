"""arXiv fetcher tests: Atom parsing, cache-skipping atomic downloads and the rate limiter.

The Atom fixture is hand-built in the arXiv API format from the real abs pages of the two papers (the API was
unavailable when it was made); it is not a byte-for-byte recording.
"""
import gzip
import io
import json
import tarfile
from pathlib import Path

import pytest

from app import arxiv
from app.arxiv import PaperMeta

FIXTURE = Path(__file__).parent / "fixtures" / "atom_two_papers.xml"


def make_tar_gz() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = br"\section{Intro}"
        info = tarfile.TarInfo("main.tex")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


@pytest.fixture
def meta() -> PaperMeta:
    return PaperMeta("1511.06279", 4, "Neural Programmer-Interpreters", ["Scott Reed"], 2015, "abstract", "cs.LG")


@pytest.fixture
def fake_http(monkeypatch):
    """Replaces the single network function; returns the list of requested URLs."""
    calls: list[str] = []
    responses = {"/e-print/": make_tar_gz(), "/pdf/": b"%PDF-1.5 fake"}

    def get(url: str) -> bytes:
        calls.append(url)
        return next(body for key, body in responses.items() if key in url)

    monkeypatch.setattr(arxiv, "_http_get", get)
    return calls


def test_parse_atom_two_entries():
    npi, gti = arxiv.parse_atom(FIXTURE.read_bytes())
    assert (npi.id, npi.version, npi.year, npi.primary_category) == ("1511.06279", 4, 2015, "cs.LG")
    assert npi.title == "Neural Programmer-Interpreters"
    assert npi.authors == ["Scott Reed", "Nando de Freitas"]
    assert npi.abstract.startswith("We propose the neural programmer-interpreter")
    assert "\n" not in npi.abstract
    assert (gti.id, gti.version, gti.year, gti.primary_category) == ("2003.06085", 2, 2020, "cs.RO")
    assert len(gti.authors) == 5 and gti.authors[-1] == "Li Fei-Fei"


def test_fetch_meta_one_batched_request_in_order(monkeypatch):
    urls = []
    monkeypatch.setattr(arxiv, "_http_get", lambda url: urls.append(url) or FIXTURE.read_bytes())
    papers = arxiv.fetch_meta(["2003.06085", "1511.06279"])
    assert [p.id for p in papers] == ["2003.06085", "1511.06279"]
    assert len(urls) == 1 and "id_list=2003.06085,1511.06279" in urls[0]


def test_fetch_meta_reports_missing_ids(monkeypatch):
    monkeypatch.setattr(arxiv, "_http_get", lambda url: FIXTURE.read_bytes())
    with pytest.raises(ValueError, match="9999.99999"):
        arxiv.fetch_meta(["1511.06279", "9999.99999"])


def test_download_writes_files(tmp_path, meta, fake_http):
    folder = arxiv.download(meta, tmp_path)
    assert folder == tmp_path / "papers" / "1511.06279"
    assert json.loads((folder / "meta.json").read_text(encoding="utf-8"))["title"] == meta.title
    assert (folder / "source.tar.gz").exists() and not (folder / "source.gz").exists()
    assert (folder / "paper.pdf").read_bytes().startswith(b"%PDF")


def test_download_single_tex_is_source_gz(tmp_path, meta, monkeypatch):
    bodies = {"/e-print/": gzip.compress(br"\documentclass{article}"), "/pdf/": b"%PDF"}
    monkeypatch.setattr(arxiv, "_http_get", lambda url: next(b for k, b in bodies.items() if k in url))
    folder = arxiv.download(meta, tmp_path)
    assert (folder / "source.gz").exists() and not (folder / "source.tar.gz").exists()


def test_download_skips_existing_files(tmp_path, meta, fake_http):
    arxiv.download(meta, tmp_path)
    fake_http.clear()
    arxiv.download(meta, tmp_path)
    assert fake_http == []

    folder = tmp_path / "papers" / meta.id
    (folder / "paper.pdf").unlink()
    arxiv.download(meta, tmp_path)
    assert len(fake_http) == 1 and "/pdf/" in fake_http[0]


def test_rate_limiter_waits_between_calls(monkeypatch):
    class FakeTime:
        now = 100.0
        slept: list[float] = []

        @classmethod
        def monotonic(cls):
            return cls.now

        @classmethod
        def sleep(cls, s):
            cls.slept.append(s)
            cls.now += s

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"ok"

    monkeypatch.setattr(arxiv, "time", FakeTime)
    monkeypatch.setattr(arxiv, "_last_request", None)
    monkeypatch.setattr(arxiv.settings, "arxiv_delay_s", 3.0)
    monkeypatch.setattr(arxiv.urllib.request, "urlopen", lambda req, timeout: FakeResponse())

    arxiv._http_get("https://example.test/a")
    assert FakeTime.slept == []  # the first call does not wait
    FakeTime.now += 1.0
    arxiv._http_get("https://example.test/b")
    assert FakeTime.slept == [pytest.approx(2.0)]
    FakeTime.now += 5.0  # already past the delay
    arxiv._http_get("https://example.test/c")
    assert len(FakeTime.slept) == 1


def test_download_is_atomic_when_the_request_fails(tmp_path, meta, monkeypatch):
    def broken(url: str) -> bytes:
        if "/pdf/" in url:
            raise OSError("connection reset")
        return make_tar_gz()

    monkeypatch.setattr(arxiv, "_http_get", broken)
    with pytest.raises(OSError):
        arxiv.download(meta, tmp_path)
    folder = tmp_path / "papers" / meta.id
    assert not (folder / "paper.pdf").exists()

    # a leftover partial write must not count as a finished file
    (folder / "paper.pdf.part").write_bytes(b"%PDF trunc")
    monkeypatch.setattr(arxiv, "_http_get", lambda url: b"%PDF full" if "/pdf/" in url else make_tar_gz())
    arxiv.download(meta, tmp_path)
    assert (folder / "paper.pdf").read_bytes() == b"%PDF full"
    assert not list(folder.glob("*.part"))


def test_pdf_only_paper_leaves_marker_and_is_not_requested_again(tmp_path, meta, monkeypatch):
    calls = []
    monkeypatch.setattr(arxiv, "_http_get", lambda url: calls.append(url) or b"%PDF-only")
    folder = arxiv.download(meta, tmp_path)
    assert (folder / "source.none").exists()
    calls.clear()
    arxiv.download(meta, tmp_path)
    assert calls == []


def test_fetch_meta_accepts_versioned_ids(monkeypatch):
    monkeypatch.setattr(arxiv, "_http_get", lambda url: FIXTURE.read_bytes())
    papers = arxiv.fetch_meta(["2003.06085v1", "1511.06279v3"])
    assert [p.id for p in papers] == ["2003.06085", "1511.06279"]


ENTRY = """<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>{id}</id><title>{title}</title>
<summary>{summary}</summary></entry></feed>"""


def test_error_entry_raises_with_summary():
    xml = ENTRY.format(id="http://arxiv.org/api/errors#incorrect_id_format_for_1234", title="Error",
                       summary="incorrect id format for 1234").encode()
    with pytest.raises(ValueError, match="incorrect id format for 1234"):
        arxiv.parse_atom(xml)


def test_old_style_id():
    xml = """<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom"><entry>
<id>http://arxiv.org/abs/cs/0112017v1</id><published>2001-12-18T00:00:00Z</published><title>Old</title>
<summary>s</summary><author><name>A B</name></author></entry></feed>""".encode()
    (paper,) = arxiv.parse_atom(xml)
    assert (paper.id, paper.version, paper.year) == ("cs/0112017", 1, 2001)
    assert arxiv.paper_dir(paper.id, Path("d")) == Path("d") / "papers" / "cs_0112017"


# ---------- retry, source detection ----------

import email.message
import urllib.error


def http_error(code, retry_after=None):
    headers = email.message.Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://arxiv.org/x", code, "err", headers, None)


@pytest.fixture
def sleeps(monkeypatch):
    slept = []
    monkeypatch.setattr(arxiv.time, "sleep", slept.append)
    return slept


@pytest.mark.parametrize("retry_after, expected", [(None, 30.0), ("7", 7.0), ("9999", 120.0), ("soon", 30.0)])
def test_get_retries_once_after_429_or_503(monkeypatch, sleeps, retry_after, expected):
    outcomes = [http_error(429, retry_after), b"body"]

    def fake(url):
        out = outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    monkeypatch.setattr(arxiv, "_http_get", fake)
    assert arxiv._get("u") == b"body"
    assert sleeps == [expected]


def test_get_gives_up_after_one_retry_and_does_not_retry_other_errors(monkeypatch, sleeps):
    calls = []

    def always(code):
        def fake(url):
            calls.append(url)
            raise http_error(code)
        return fake

    monkeypatch.setattr(arxiv, "_http_get", always(503))
    with pytest.raises(urllib.error.HTTPError):
        arxiv._get("u")
    assert len(calls) == 2 and sleeps == [30.0]
    calls.clear()
    monkeypatch.setattr(arxiv, "_http_get", always(404))
    with pytest.raises(urllib.error.HTTPError):
        arxiv._get("u")
    assert len(calls) == 1


def test_tar_is_detected_without_decompressing_the_whole_payload(tmp_path, meta, monkeypatch):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"x" * 5_000_000
        info = tarfile.TarInfo("big.tex")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    monkeypatch.setattr(gzip, "decompress", lambda *a: pytest.fail("whole payload decompressed"))
    monkeypatch.setattr(arxiv, "_http_get", lambda url: buf.getvalue() if "/e-print/" in url else b"%PDF")
    folder = arxiv.download(meta, tmp_path)
    assert (folder / "source.tar.gz").exists()


@pytest.mark.parametrize("payload", [b"<html>Too many requests</html>", b"", b"\x1f\x8bgarbage"])
def test_non_gzip_non_pdf_source_raises_and_writes_nothing(tmp_path, meta, monkeypatch, payload):
    monkeypatch.setattr(arxiv, "_http_get", lambda url: payload)
    with pytest.raises(OSError):
        arxiv.download(meta, tmp_path)
    folder = tmp_path / "papers" / meta.id
    assert not any(folder.glob("source.*")) and not (folder / "paper.pdf").exists()
