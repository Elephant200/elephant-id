"""Parsing and formatting of SEEK ear codes for elephant identification."""

import re
from dataclasses import dataclass
from typing import Literal

Gender = Literal["B", "C"]
RightEarSector = Literal[0, 7, 8, 9]
LeftEarSector = Literal[0, 3, 4, 5]

_RIGHT_EAR_SECTORS: frozenset[int] = frozenset({0, 7, 8, 9})
_LEFT_EAR_SECTORS: frozenset[int] = frozenset({0, 3, 4, 5})

# Codec lookup tables. Dict lookups are faster and clearer than helper methods.
_BOOL_FROM_CHAR: dict[str, bool | None] = {"_": None, "0": False, "1": True}
_BOOL_TO_CHAR: dict[bool | None, str] = {None: "_", False: "0", True: "1"}
_SECTOR_FROM_CHAR: dict[str, int | None] = {
    "_": None, "0": 0, "3": 3, "4": 4, "5": 5, "7": 7, "8": 8, "9": 9,
}
_SECTOR_TO_CHAR: dict[int | None, str] = {
    None: "_", 0: "0", 3: "3", 4: "4", 5: "5", 7: "7", 8: "8", 9: "9",
}

# Compiled once
_SEEK_RE = re.compile(
    r"(?P<g>[BC_])"
    r"(?P<a>\d{2}|__)"
    r"T(?P<tr>[01_])(?P<tl>[01_])"
    r"E(?P<rt1>[0789_])(?P<rh1>[0789_])(?P<rt2>[0789_])(?P<rh2>[0789_])"
    r"-(?P<lt1>[0345_])(?P<lh1>[0345_])(?P<lt2>[0345_])(?P<lh2>[0345_])"
    r"X(?P<xr>[01_])(?P<xl>[01_])"
    r"S(?P<sr>[01_])(?P<sl>[01_])(?P<sb>[01_])"
)


@dataclass(frozen=True, slots=True)
class SeekCode:
    """Structured representation of a SEEK code.

    A SEEK code is a fixed-width string that encodes coarse, human-observable
    features of an elephant. The grammar accepted by :meth:`from_str` is::

        <g> <aa> T<r><l> E<rt1><rh1><rt2><rh2>-<lt1><lh1><lt2><lh2> X<r><l> S<r><l><b>

    where each component is:

    * ``<g>``        — gender: ``B`` (bull), ``C`` (cow), or ``_`` (unknown).
    * ``<aa>``       — birth-year (last two digits), or ``__`` if unknown.
    * ``T<r><l>``  — right/left tusk presence: ``0``, ``1``, or ``_``.
    * ``E<rt1><rh1><rt2><rh2>-<lt1><lh1><lt2><lh2>``       — four right-ear sectors: largest tear, largest hole,
                     second-largest tear, second-largest hole. Each is one of
                     ``0/7/8/9`` (right-ear sector ids) or ``_``.
    * ``-<lt1><lh1><lt2><lh2>``       — four left-ear sectors with the same meaning, using the
                     left-ear sector ids ``0/3/4/5`` or ``_``.
    * ``X<r><l>``  — right/left "extreme feature" flags.
    * ``S<r><l><b>`` — special features on right ear / left ear / body.

    Instances are immutable, hashable, and use ``__slots__``, so they can
    be held in large quantities and used as dict keys / set members.
    Use :meth:`from_str` to parse a code and ``str(code)`` to format one;
    round-tripping is exact.

    All fields are ``None`` when unknown.
    """

    g: Gender | None  # Gender of the elephant. Bull or cow.
    a: int | None  # Age (birth year, last two digits) of the elephant. Very approximate.

    tr: bool | None  # Whether the elephant has a right tusk.
    tl: bool | None  # Whether the elephant has a left tusk.

    rt1: RightEarSector | None = None  # Sector of largest tear on the right ear.
    rh1: RightEarSector | None = None  # Sector of largest hole on the right ear.
    rt2: RightEarSector | None = None  # Sector of second largest tear on the right ear.
    rh2: RightEarSector | None = None  # Sector of second largest hole on the right ear.

    lt1: LeftEarSector | None = None  # Sector of largest tear on the left ear.
    lh1: LeftEarSector | None = None  # Sector of largest hole on the left ear.
    lt2: LeftEarSector | None = None  # Sector of second largest tear on the left ear.
    lh2: LeftEarSector | None = None  # Sector of second largest hole on the left ear.

    xr: bool | None = None  # Whether the elephant has a right extreme feature.
    xl: bool | None = None  # Whether the elephant has a left extreme feature.

    sr: bool | None = None  # Whether the elephant has a special feature on the right ear.
    sl: bool | None = None  # Whether the elephant has a special feature on the left ear.
    sb: bool | None = None  # Whether the elephant has a special feature on the body.

    def __post_init__(self) -> None:
        """Validate field invariants at construction time.

        Runs for every instance regardless of whether it was built via
        :meth:`from_str` or by direct keyword construction. Raises
        :class:`ValueError` on the first violation found.
        """
        if self.g is not None and self.g not in ("B", "C"):
            raise ValueError(f"Invalid gender: {self.g!r}")
        if self.a is not None and not 0 <= self.a <= 99:
            raise ValueError(f"Age out of range (0..99): {self.a}")
        for name, v in (("rt1", self.rt1), ("rh1", self.rh1),
                        ("rt2", self.rt2), ("rh2", self.rh2)):
            if v is not None and v not in _RIGHT_EAR_SECTORS:
                raise ValueError(
                    f"{name}={v!r} is not a right-ear sector "
                    f"(allowed: {sorted(_RIGHT_EAR_SECTORS)})"
                )
        for name, v in (("lt1", self.lt1), ("lh1", self.lh1),
                        ("lt2", self.lt2), ("lh2", self.lh2)):
            if v is not None and v not in _LEFT_EAR_SECTORS:
                raise ValueError(
                    f"{name}={v!r} is not a left-ear sector "
                    f"(allowed: {sorted(_LEFT_EAR_SECTORS)})"
                )

    @classmethod
    def from_str(cls, code: str) -> "SeekCode":
        """Parse a SEEK code string into a :class:`SeekCode`.

        Args:
            code: The SEEK code, exactly matching the grammar in
                :class:`SeekCode`.

        Returns:
            A new :class:`SeekCode` populated from ``code``.

        Raises:
            ValueError: If ``code`` does not match the grammar (wrong
                length, illegal sector for a side, unrecognised gender,
                etc.).
        """
        match = _SEEK_RE.fullmatch(code)
        if match is None:
            raise ValueError(f"Invalid SEEK code: {code}")

        parts = match.groupdict()
        boolean = _BOOL_FROM_CHAR
        sector = _SECTOR_FROM_CHAR
        return cls(
            g=None if parts["g"] == "_" else parts["g"],
            a=None if parts["a"] == "__" else int(parts["a"]),
            tr=boolean[parts["tr"]], tl=boolean[parts["tl"]],
            rt1=sector[parts["rt1"]], rh1=sector[parts["rh1"]],
            rt2=sector[parts["rt2"]], rh2=sector[parts["rh2"]],
            lt1=sector[parts["lt1"]], lh1=sector[parts["lh1"]],
            lt2=sector[parts["lt2"]], lh2=sector[parts["lh2"]],
            xr=boolean[parts["xr"]], xl=boolean[parts["xl"]],
            sr=boolean[parts["sr"]], sl=boolean[parts["sl"]], sb=boolean[parts["sb"]],
        )

    def __str__(self) -> str:
        """Render this code in the canonical SEEK string form.

        ``SeekCode.from_str(str(code)) == code`` for any valid instance.
        """
        boolean = _BOOL_TO_CHAR
        sector = _SECTOR_TO_CHAR
        age = "__" if self.a is None else f"{self.a:02d}"
        return (
            f"{self.g or '_'}{age}"
            f"T{boolean[self.tr]}{boolean[self.tl]}"
            f"E{sector[self.rt1]}{sector[self.rh1]}{sector[self.rt2]}{sector[self.rh2]}"
            f"-{sector[self.lt1]}{sector[self.lh1]}{sector[self.lt2]}{sector[self.lh2]}"
            f"X{boolean[self.xr]}{boolean[self.xl]}"
            f"S{boolean[self.sr]}{boolean[self.sl]}{boolean[self.sb]}"
        )
