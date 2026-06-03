"""Filter configuration and per-sighting matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import seek_codes

if TYPE_CHECKING:
    from .state import SightingKey


def _year_from_date(date_str: str) -> int | None:
    try:
        return int(date_str.strip().split("-", 1)[0])
    except (ValueError, IndexError):
        return None


def _coerce_optional_int(val: object) -> int | None:
    if val is None or val == "":
        return None
    try:
        return int(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _birth_year_midpoint(decade: int, sighting_year: int) -> int:
    """Full birth year assuming birth at the recorded decade's midpoint.

    ``decade`` is the last two digits of the birth decade as recorded in the
    SEEK code (e.g. ``80`` → the 1980s). The century is the most recent one
    whose decade start does not exceed ``sighting_year`` — so ``0`` reads as
    the 2000s rather than the 1900s for the dataset's 2002+ sightings.
    """
    starts = [base + decade for base in (1900, 2000) if base + decade <= sighting_year]
    start = max(starts) if starts else 1900 + decade
    return start + 5


def age_from_decade(decade: int | None, sighting_year: int | None) -> int | None:
    """Approximate age at sighting, or ``None`` when either input is unknown.

    The birth year is taken as the midpoint of the recorded birth decade, so
    the result is coarse. Clamped at ``0`` since a midpoint estimate can land
    just after a sighting (e.g. a calf seen early in its own birth decade).
    """
    if decade is None or sighting_year is None:
        return None
    return max(0, sighting_year - _birth_year_midpoint(decade, sighting_year))


@dataclass
class FilterConfig:
    sex_bull: bool = False
    sex_cow: bool = False
    tusk_both: bool = False
    tusk_left_only: bool = False
    tusk_right_only: bool = False
    tusk_no_tusks: bool = False
    extreme_left: bool = False
    extreme_right: bool = False
    special_left_ear: bool = False
    special_right_ear: bool = False
    special_body: bool = False
    non_normal_only: bool = False
    year_min: int | None = None
    year_max: int | None = None
    age_min: int | None = None
    age_max: int | None = None

    @classmethod
    def from_json(cls, data: dict) -> FilterConfig:
        sex = data.get("sex") or {}
        tusks = data.get("tusks") or {}
        ex = data.get("extreme") or {}
        sp = data.get("special") or {}
        return cls(
            sex_bull=bool(sex.get("B")),
            sex_cow=bool(sex.get("C")),
            tusk_both=bool(tusks.get("both")),
            tusk_left_only=bool(tusks.get("leftOnly")),
            tusk_right_only=bool(tusks.get("rightOnly")),
            tusk_no_tusks=bool(tusks.get("noTusks")),
            extreme_left=bool(ex.get("left")),
            extreme_right=bool(ex.get("right")),
            special_left_ear=bool(sp.get("leftEar")),
            special_right_ear=bool(sp.get("rightEar")),
            special_body=bool(sp.get("body")),
            non_normal_only=bool(data.get("nonNormalOnly")),
            year_min=_coerce_optional_int(data.get("yearMin")),
            year_max=_coerce_optional_int(data.get("yearMax")),
            age_min=_coerce_optional_int(data.get("ageMin")),
            age_max=_coerce_optional_int(data.get("ageMax")),
        )

    def to_json(self) -> dict:
        return {
            "sex": {"B": self.sex_bull, "C": self.sex_cow},
            "tusks": {
                "both": self.tusk_both,
                "leftOnly": self.tusk_left_only,
                "rightOnly": self.tusk_right_only,
                "noTusks": self.tusk_no_tusks,
            },
            "extreme": {
                "left": self.extreme_left,
                "right": self.extreme_right,
            },
            "special": {
                "leftEar": self.special_left_ear,
                "rightEar": self.special_right_ear,
                "body": self.special_body,
            },
            "nonNormalOnly": self.non_normal_only,
            "yearMin": self.year_min,
            "yearMax": self.year_max,
            "ageMin": self.age_min,
            "ageMax": self.age_max,
        }

    def sex_active(self) -> bool:
        return self.sex_bull or self.sex_cow

    def tusk_active(self) -> bool:
        return (
            self.tusk_both
            or self.tusk_left_only
            or self.tusk_right_only
            or self.tusk_no_tusks
        )

    def extreme_active(self) -> bool:
        return self.extreme_left or self.extreme_right

    def special_active(self) -> bool:
        return self.special_left_ear or self.special_right_ear or self.special_body

    def years_active(self) -> bool:
        return self.year_min is not None or self.year_max is not None

    def ages_active(self) -> bool:
        return self.age_min is not None or self.age_max is not None


def matches(
    key: SightingKey,
    elephant_seek: dict[str, str],
    cfg: FilterConfig,
) -> bool:
    """Return whether ``key`` survives all active filters in ``cfg``."""
    if cfg.years_active():
        y = _year_from_date(key.date)
        if y is None:
            return False
        if cfg.year_min is not None and y < cfg.year_min:
            return False
        if cfg.year_max is not None and y > cfg.year_max:
            return False

    code = elephant_seek.get(key.name, "")
    parsed = seek_codes.parse(code)

    if cfg.ages_active():
        age = age_from_decade(parsed.age, _year_from_date(key.date))
        if age is None:
            return False
        if cfg.age_min is not None and age < cfg.age_min:
            return False
        if cfg.age_max is not None and age > cfg.age_max:
            return False

    if cfg.sex_active() and not (
        (cfg.sex_bull and parsed.sex == "B")
        or (cfg.sex_cow and parsed.sex == "C")
    ):
        return False

    if cfg.tusk_active() and not (
        (cfg.tusk_both and parsed.tusk_left == "1" and parsed.tusk_right == "1")
        or (cfg.tusk_left_only and parsed.tusk_left == "1" and parsed.tusk_right == "0")
        or (cfg.tusk_right_only and parsed.tusk_left == "0" and parsed.tusk_right == "1")
        or (cfg.tusk_no_tusks and parsed.tusk_left == "0" and parsed.tusk_right == "0")
    ):
        return False

    if cfg.extreme_active() and not (
        (cfg.extreme_left and parsed.extreme_left == 1)
        or (cfg.extreme_right and parsed.extreme_right == 1)
    ):
        return False

    if cfg.special_active() and not (
        (cfg.special_left_ear and parsed.special_left == 1)
        or (cfg.special_right_ear and parsed.special_mid == 1)
        or (cfg.special_body and parsed.special_right == 1)
    ):
        return False

    if cfg.non_normal_only and not seek_codes.explicit_non_normal(code):
        return False

    return True


year_from_date = _year_from_date
