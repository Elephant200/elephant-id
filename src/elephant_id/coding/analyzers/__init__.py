"""Per-photo field analyzers.

Each analyzer is a class constructed once with its dependencies and config,
then called as ``analyze(photo, prep)`` per photo, returning a loose dict (or
list) of field evidence. Evidence is kept untyped for now; typed structures
and review flags come later.
"""

from .age import AgeAnalyzer
from .ears import EarAnalyzer
from .gender import GenderAnalyzer
from .tusks import TuskAnalyzer

__all__ = ["AgeAnalyzer", "EarAnalyzer", "GenderAnalyzer", "TuskAnalyzer"]
