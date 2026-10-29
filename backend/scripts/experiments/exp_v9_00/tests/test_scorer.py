"""The §19 neutral scorer on hand-built sessions (no private data)."""

import math

import pytest
from capacity import predictive, target_context
from packing import NONE as PACKING_NONE
from scipy import stats
from scorer import RULES, chain, lower_log_score, predictor, upper_bound, verdict, width80
from test_learning import rest_corpus, session, st


def test_one_fit_per_session_matches_capacity_predictive():
    sessions = rest_corpus()
    probe = rest_corpus(1)[0]
    probe["startedAt"] = probe["endedAt"] = "2026-10-01T18:00:00Z"
    for rule in RULES:
        at = predictor(rule, sessions)
        for i in (1, 3):
            ctx = target_context(probe, i, sessions)
            for v in ("full", "rv"):
                assert at(v, ctx) == pytest.approx(predictive(rule, sessions, v, ctx))
    assert predictor("v2", [])("full", {}) is None


class PriorOnly:
    """A diver with a physiology prior and nothing else, so v5 has a posterior before any row."""

    packing = PACKING_NONE

    def vo2(self, sessions, lag=True):
        return {"vo2": 200.0}

    def prior_at(self, vo2):
        return {
            "full": (math.log(300), 0.2),
            "frc": (math.log(150), 0.2),
            "rv": (math.log(100), 0.2),
        }


def test_v5_with_a_prior_and_no_rows_has_no_predictive_where_capacity_raises():
    train = [session([st("hold", 200, style="full")])]  # no trace, so no v5 row
    ctx = {"holdIndex": 0.0, "logRestBefore": 0.0}
    with pytest.raises(ValueError):
        predictive("v5", train, "full", ctx, PriorOnly())
    assert predictor("v5", train, PriorOnly())("full", ctx) is None


def test_scores_on_a_hand_computed_predictive():
    p = (math.log(300), 0.2, 1e7)  # df this large is the normal
    q = stats.norm.ppf(0.9)
    assert lower_log_score(300, p) == pytest.approx(math.log(0.5))
    assert upper_bound(p) == pytest.approx(300 * math.exp(0.2 * q), rel=1e-6)
    assert width80(p) == pytest.approx(300 * (math.exp(0.2 * q) - math.exp(-0.2 * q)), rel=1e-6)


def m(ls, share, width):
    return {"ls": ls, "share": share, "width": width}


def test_a_rule_is_preferred_only_when_it_holds_on_all_three():
    assert verdict({"v2": m(-0.5, 0.1, 200), "v5": m(-0.4, 0.1, 200)}) == "v5"
    assert verdict({"v2": m(-0.5, 0.0, 200), "v5": m(-0.4, 0.1, 200)}) == "undecided"
    assert verdict({"v2": m(-0.5, 0.1, 150), "v5": m(-0.4, 0.1, 200)}) == "undecided"
    assert verdict({"v2": m(-0.3, 0.0, 150), "v5": m(-0.4, 0.1, 200)}) == "v2"


def test_chain_scores_each_session_on_the_sessions_before_it():
    records, skipped = chain(rest_corpus(4))
    assert skipped["full"] == ["2026-09-10"]  # nothing came before the first session
    assert [r["date"][:10] for r in records["full"]] == ["2026-09-11", "2026-09-12", "2026-09-13"]
    for r in records["full"]:
        assert len(r["ls"]["v2"]) == len(r["ls"]["v5"]) == len(r["actual"]) == 2
