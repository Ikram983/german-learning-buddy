import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from german_buddy.tools import MistakeTracker, conjugate  # noqa: E402


def test_conjugate_regular_verb():
    forms = conjugate("machen")
    assert forms["ich"] == "mache"
    assert forms["du"] == "machst"
    assert forms["wir"] == "machen"


def test_conjugate_regular_verb_with_stem_ending_in_t():
    # arbeiten needs an extra 'e' before -st/-t endings
    forms = conjugate("arbeiten")
    assert forms["du"] == "arbeitest"
    assert forms["er/sie/es"] == "arbeitet"


def test_conjugate_regular_verb_with_sibilant_stem():
    # reisen: du-form should be "reist", not "reisst"
    forms = conjugate("reisen")
    assert forms["du"] == "reist"


def test_conjugate_irregular_verb():
    forms = conjugate("fahren")
    assert forms["du"] == "fährst"
    assert forms["er/sie/es"] == "fährt"


def test_conjugate_sein_fully_irregular():
    forms = conjugate("sein")
    assert forms == {
        "ich": "bin",
        "du": "bist",
        "er/sie/es": "ist",
        "wir": "sind",
        "ihr": "seid",
        "sie/Sie": "sind",
    }


def test_conjugate_rejects_non_infinitive():
    import pytest

    with pytest.raises(ValueError):
        conjugate("Haus")  # not a verb infinitive


def test_mistake_tracker_counts_and_ranks(tmp_path):
    tracker = MistakeTracker(tmp_path / "mistakes.json")
    tracker.record_mistake("accusative_case")
    tracker.record_mistake("accusative_case")
    tracker.record_mistake("verb:fahren")

    top = tracker.top_mistakes(n=2)
    assert top[0] == ("accusative_case", 2)
    assert top[1] == ("verb:fahren", 1)


def test_mistake_tracker_persists_across_instances(tmp_path):
    path = tmp_path / "mistakes.json"
    MistakeTracker(path).record_mistake("dative_case")

    reloaded = MistakeTracker(path)
    assert reloaded.top_mistakes() == [("dative_case", 1)]
