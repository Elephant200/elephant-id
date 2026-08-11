"""Tests for :class:`PhotoAnalyzer` orchestration seams."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from elephant_id.coding.photo_analyzer import PhotoAnalyzer
from elephant_id.domain import Photo


@pytest.fixture
def stubbed_analyzer(monkeypatch: pytest.MonkeyPatch) -> tuple[PhotoAnalyzer, list[str]]:
    """A PhotoAnalyzer whose detection pipeline is stubbed and evidence spied.

    The detection helpers are replaced with sentinels so no model or image is
    touched, and each evidence analyzer records its name when invoked. Returns
    the analyzer and the shared call log.
    """
    analyzer = PhotoAnalyzer(dataset=object())  # type: ignore[arg-type]

    monkeypatch.setattr(analyzer.sam3, "run", lambda photo, preset: ["detection"])
    monkeypatch.setattr(analyzer, "_choose_body", lambda photo, dets: "BODY")
    monkeypatch.setattr(analyzer, "_features_on_body", lambda body, feats: feats)
    monkeypatch.setattr(
        analyzer, "_group_features", lambda photo, feats: (["TRUNK"], ["EAR"], ["TUSK"])
    )
    monkeypatch.setattr(analyzer, "_choose_usable_ears", lambda photo, ears: ears)
    monkeypatch.setattr(analyzer, "_anchor_ears", lambda photo, ears: ["ANCHORED_EAR"])
    monkeypatch.setattr(analyzer, "_estimate_view", lambda **kwargs: "left")

    calls: list[str] = []
    for name in ("age_analyzer", "gender_analyzer", "ear_analyzer", "tusk_analyzer"):
        field_analyzer = getattr(analyzer, name)
        monkeypatch.setattr(
            field_analyzer,
            "analyze",
            lambda photo, ctx, _name=name: calls.append(_name) or {"field": _name},
        )
    return analyzer, calls


def test_analyze_shared_skips_evidence_analyzers(
    stubbed_analyzer: tuple[PhotoAnalyzer, list[str]],
    make_photo: Callable[..., Photo],
) -> None:
    analyzer, calls = stubbed_analyzer

    shared = analyzer.analyze_shared(make_photo())

    assert shared == {
        "view": "left",
        "body": "BODY",
        "trunks": ["TRUNK"],
        "ears": ["ANCHORED_EAR"],
        "tusks": ["TUSK"],
    }
    assert calls == []  # none of the evidence analyzers ran


def test_analyze_runs_evidence_on_the_shared_result(
    stubbed_analyzer: tuple[PhotoAnalyzer, list[str]],
    make_photo: Callable[..., Photo],
) -> None:
    analyzer, calls = stubbed_analyzer

    result = analyzer.analyze(make_photo())

    assert result is not None
    assert result["view"] == "left"
    assert result["shared_data"] == {
        "body": "BODY",
        "trunks": ["TRUNK"],
        "ears": ["ANCHORED_EAR"],
        "tusks": ["TUSK"],
    }
    assert set(calls) == {"age_analyzer", "gender_analyzer", "ear_analyzer", "tusk_analyzer"}


def test_analyze_returns_none_when_shared_is_none(
    stubbed_analyzer: tuple[PhotoAnalyzer, list[str]],
    make_photo: Callable[..., Photo],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analyzer, calls = stubbed_analyzer
    monkeypatch.setattr(analyzer.sam3, "run", lambda photo, preset: [])

    assert analyzer.analyze(make_photo()) is None
    assert calls == []
