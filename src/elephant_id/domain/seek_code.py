"""Parsing and formatting of SEEK ear codes for elephant identification."""

import re
from dataclasses import dataclass
from typing import Literal

type Gender = Literal["B", "C"]
type RightEarSector = Literal[0, 7, 8, 9]
type LeftEarSector = Literal[0, 3, 4, 5]

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
    r"E(?P<ert1>[0789_])(?P<erh1>[0789_])(?P<ert2>[0789_])(?P<erh2>[0789_])"
    r"-(?P<elt1>[0345_])(?P<elh1>[0345_])(?P<elt2>[0345_])(?P<elh2>[0345_])"
    r"X(?P<xr>[01_])(?P<xl>[01_])"
    r"S(?P<sr>[01_])(?P<sl>[01_])(?P<sb>[01_])"
)


@dataclass(frozen=True, slots=True)
class SeekCode:
    """Structured representation of a SEEK code.

    A SEEK code is a fixed-width string that encodes coarse, human-observable
    features of an elephant. The grammar accepted by :meth:`from_str` is::

        <g> <aa> T<r><l> E<rt1><rh1><rt2><rh2>-<lt1><lh1><lt2><lh2> X<r><l> S<r><l><b>

    The spaces above are only for readability; an actual code is the components
    concatenated with no separators (the literal markers ``T``/``E``/``X``/``S``
    and the ``-`` between the ear groups are required). Where each component is:

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

    ert1: RightEarSector | None = None  # Sector of largest tear on the right ear.
    erh1: RightEarSector | None = None  # Sector of largest hole on the right ear.
    ert2: RightEarSector | None = None  # Sector of second largest tear on the right ear.
    erh2: RightEarSector | None = None  # Sector of second largest hole on the right ear.

    elt1: LeftEarSector | None = None  # Sector of largest tear on the left ear.
    elh1: LeftEarSector | None = None  # Sector of largest hole on the left ear.
    elt2: LeftEarSector | None = None  # Sector of second largest tear on the left ear.
    elh2: LeftEarSector | None = None  # Sector of second largest hole on the left ear.

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
        if self.a is not None and (type(self.a) is not int or not 0 <= self.a <= 99):
            raise ValueError(f"Invalid age (expected int 0..99): {self.a!r}")
        for name, v in (("tr", self.tr), ("tl", self.tl),
                        ("xr", self.xr), ("xl", self.xl),
                        ("sr", self.sr), ("sl", self.sl), ("sb", self.sb)):
            if v is not None and not isinstance(v, bool):
                raise ValueError(f"{name}={v!r} is not a boolean flag")
        for name, v in (("ert1", self.ert1), ("erh1", self.erh1),
                        ("ert2", self.ert2), ("erh2", self.erh2)):
            if v is not None and (type(v) is not int or v not in _RIGHT_EAR_SECTORS):
                raise ValueError(
                    f"{name}={v!r} is not a right-ear sector "
                    f"(allowed: {sorted(_RIGHT_EAR_SECTORS)})"
                )
        for name, v in (("elt1", self.elt1), ("elh1", self.elh1),
                        ("elt2", self.elt2), ("elh2", self.elh2)):
            if v is not None and (type(v) is not int or v not in _LEFT_EAR_SECTORS):
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
            ert1=sector[parts["ert1"]], erh1=sector[parts["erh1"]],
            ert2=sector[parts["ert2"]], erh2=sector[parts["erh2"]],
            elt1=sector[parts["elt1"]], elh1=sector[parts["elh1"]],
            elt2=sector[parts["elt2"]], elh2=sector[parts["elh2"]],
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
            f"E{sector[self.ert1]}{sector[self.erh1]}{sector[self.ert2]}{sector[self.erh2]}"
            f"-{sector[self.elt1]}{sector[self.elh1]}{sector[self.elt2]}{sector[self.elh2]}"
            f"X{boolean[self.xr]}{boolean[self.xl]}"
            f"S{boolean[self.sr]}{boolean[self.sl]}{boolean[self.sb]}"
        )
