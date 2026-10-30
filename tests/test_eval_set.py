"""The labelled question set: well formed always; labels checked against the parsed papers when they are downloaded."""
from collections import Counter

import pytest

from app.config import ROOT
from app.evaluate import QUESTIONS, load_questions, section_matches
from app.sections import load_sections

READING_LIST = {"1511.06279", "1710.01813", "1803.01840", "2003.06085", "2206.11795"}
PAPERS = ROOT / "data" / "papers"  # the real downloads, not the per-test data folder
MAX_PATHS_PER_SUBSTRING = 3  # a label names a section, not a whole chapter of the paper


def test_question_set_is_well_formed():
    questions = load_questions(QUESTIONS)
    assert 30 <= len(questions) <= 32
    assert len({q.id for q in questions}) == len(questions)
    for q in questions:
        assert q.question.strip() and q.papers and 1 <= len(q.keywords) <= 3, q.id
        assert set(q.papers) <= READING_LIST, q.id
        assert all(q.gold_sections(pid) for pid in q.papers), q.id  # every gold paper names its sections
        assert q.type in {"factual", "method", "term", "cross"}, q.id
        assert (q.type == "cross") == (len(q.papers) > 1), q.id
        assert isinstance(q.sections, dict) == (len(q.papers) > 1), q.id
    assert sum(q.type == "cross" for q in questions) >= 6
    per_paper = Counter(q.papers[0] for q in questions if q.type != "cross")
    assert set(per_paper) == READING_LIST and min(per_paper.values()) >= 4


@pytest.fixture(scope="module")
def sections():
    if not all((PAPERS / pid / "meta.json").is_file() for pid in READING_LIST):
        pytest.skip("reading list not downloaded under data/papers")
    return {pid: load_sections(PAPERS / pid) for pid in READING_LIST}


def test_gold_labels_match_the_parsed_papers(sections):
    problems = []
    for q in load_questions(QUESTIONS):
        gold = []
        for pid in q.papers:
            mine = []
            for sub in q.gold_sections(pid):
                matched = [s for s in sections[pid] if section_matches(s.path, [sub])]
                if not matched:
                    problems.append(f"{q.id}: no section of {pid} is titled with {sub!r}")
                if len(matched) > MAX_PATHS_PER_SUBSTRING:
                    problems.append(f"{q.id}: {sub!r} matches {len(matched)} sections of {pid}; be more specific")
                mine += matched
            if not any(kw.lower() in s.text.lower() for s in mine for kw in q.keywords):
                problems.append(f"{q.id}: no gold section of {pid} contains a keyword")
            gold += mine
        for kw in q.keywords:
            if not any(kw.lower() in s.text.lower() for s in gold):
                problems.append(f"{q.id}: keyword {kw!r} not in its gold sections")
    assert not problems, "\n".join(problems)


def test_keywords_do_not_echo_the_question():
    echoes = [f"{q.id}: {kw!r}" for q in load_questions(QUESTIONS) for kw in q.keywords
              if kw.lower() in q.question.lower()]
    assert not echoes, "\n".join(echoes)
