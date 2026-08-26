"""Parsing helpers for SEEK codes.

A SEEK code is a compact string describing an elephant. The leading
character may be a legacy digit, which we strip. After that:

* index 0:        sex (`B` bull / `C` cow)
* indices 1..2:   `<aa>` age (birth-year last two digits), or `__`
* indices 3..6:   `T<left><right>` for tusks (`0`/`1`)
* trailing tail:  `X<xl><xr>S<sl><sm><sr>` for extreme/special
                  features; other characters there are wildcards.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Optional trailing `_` … suffix is common for unknown tail; keep it on the
# string and match it here.
TUSK_XS_RE = re.compile(r"X(.)(.)S(...)(_*)$")


@dataclass(frozen=True)
class ParsedSeek:
    sex: str | None
    age: int | None
    tusk_left: str | None
    tusk_right: str | None
    has_xs: bool
    extreme_left: int | None
    extreme_right: int | None
    special_left: int | None
    special_mid: int | None
    special_right: int | None

    @property
    def tusks_known(self) -> bool:
        return self.tusk_left is not None and self.tusk_right is not None


def strip_leading_digit(code: str) -> str:
    c = (code or "").strip()
    if c and c[0].isdigit():
        return c[1:]
    return c


def slot_01(ch: str) -> int | None:
    if ch == "0":
        return 0
    if ch == "1":
        return 1
    return None


def explicit_non_normal(code_raw: str) -> bool:
    """True iff SEEK explicitly flags non-normal.

    A code is non-normal when either T slot is `0`, or any X/S slot is
    `1`. Other characters there are treated as unknown.
    """
    s = strip_leading_digit((code_raw or "").strip())
    if not s:
        return False

    if len(s) >= 6 and s[3] == "T" and (s[4] == "0" or s[5] == "0"):
        return True

    xm = TUSK_XS_RE.search(s)
    if xm:
        if xm.group(1) == "1" or xm.group(2) == "1":
            return True
        trip = xm.group(3)
        if len(trip) == 3 and any(c == "1" for c in trip):
            return True

    return False


def parse(code_raw: str) -> ParsedSeek:
    """Parse a SEEK code into the fields the filter cares about."""
    s = strip_leading_digit(code_raw or "")

    sex: str | None = None
    if s and s[0] in ("B", "C"):
        sex = s[0]

    age: int | None = None
    if len(s) >= 3 and s[1:3].isdigit():
        age = int(s[1:3])

    tusk_left: str | None = None
    tusk_right: str | None = None
    if len(s) >= 6 and s[3] == "T":
        a, b = s[4], s[5]
        if a in "01" and b in "01":
            tusk_left, tusk_right = a, b

    xm = TUSK_XS_RE.search(s)
    extreme_left = extreme_right = None
    special_left = special_mid = special_right = None
    if xm:
        extreme_left = slot_01(xm.group(1))
        extreme_right = slot_01(xm.group(2))
        trip = xm.group(3)
        special_left = slot_01(trip[0])
        special_mid = slot_01(trip[1])
        special_right = slot_01(trip[2])

    return ParsedSeek(
        sex=sex,
        age=age,
        tusk_left=tusk_left,
        tusk_right=tusk_right,
        has_xs=xm is not None,
        extreme_left=extreme_left,
        extreme_right=extreme_right,
        special_left=special_left,
        special_mid=special_mid,
        special_right=special_right,
    )
