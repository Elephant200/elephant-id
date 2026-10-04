"""Implementation-independent identity-retrieval evaluation."""

from .benchmark import Benchmark, load_benchmark
from .evaluator import EvaluationResult, Scores, evaluate

__all__ = ["Benchmark", "EvaluationResult", "Scores", "evaluate", "load_benchmark"]
