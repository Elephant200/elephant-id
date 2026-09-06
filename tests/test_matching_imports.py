"""Import isolation for the shared matching and saved-score evaluation contracts."""

import subprocess
import sys


def test_saved_score_evaluation_does_not_import_matcher_or_model_implementations() -> (
    None
):
    """A fresh interpreter can audit scores without loading an algorithm stack."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from uuid import UUID
from elephant_id.matching import CatalogMatcher
from elephant_id.evaluation import EvaluationResult
from elephant_id.evaluation.comparison import paired_delta
scores = {"a": {UUID(int=1): {"a": 1.0, "b": 0.0}}}
assert EvaluationResult(scores).metrics["top_1"] == 1.0
assert paired_delta(scores, scores, resamples=10) == (0.0, (0.0, 0.0))
for prefix in (
    "elephant_id.matching.alphaphant", "elephant_id.matching.curvrank",
    "elephant_id.matching.miewid", "elephant_id.preparation",
    "elephant_id.inference", "torch", "ultralytics", "transformers", "inference_sdk",
):
    assert not any(name == prefix or name.startswith(prefix + ".") for name in sys.modules), prefix
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
