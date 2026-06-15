import dataclasses

import pytest

from elephant_id.domain import SeekCode


def test_seek_code_from_str_all_zero():
    code = SeekCode.from_str("B00T00E0000-0000X00S000")
    assert code.g == "B"
    assert code.a == 0
    assert code.tr is False and code.tl is False
    assert code.ert1 == 0 and code.erh1 == 0 and code.ert2 == 0 and code.erh2 == 0
    assert code.elt1 == 0 and code.elh1 == 0 and code.elt2 == 0 and code.elh2 == 0
    assert code.xr is False and code.xl is False
    assert code.sr is False and code.sl is False and code.sb is False


def test_seek_code_from_str_all_unknown():
    code = SeekCode.from_str("___T__E____-____X__S___")
    for v in (code.g, code.a, code.tr, code.tl,
              code.ert1, code.erh1, code.ert2, code.erh2,
              code.elt1, code.elh1, code.elt2, code.elh2,
              code.xr, code.xl, code.sr, code.sl, code.sb):
        assert v is None


def test_seek_code_to_str():
    code = SeekCode(
        g="C", a=0, tr=False, tl=False,
        ert1=0, erh1=0, ert2=0, erh2=0, elt1=0, elh1=0, elt2=0, elh2=0,
        xr=False, xl=False, sr=False, sl=False, sb=False,
    )
    assert str(code) == "C00T00E0000-0000X00S000"

def test_seek_code_to_str_repr():
    code = SeekCode(
        g="C", a=0, tr=False, tl=False,
        ert1=0, erh1=0, ert2=0, erh2=0, elt1=0, elh1=0, elt2=0, elh2=0,
        xr=False, xl=False, sr=False, sl=False, sb=False,
    )
    assert str(code) == "C00T00E0000-0000X00S000"
    assert repr(code) == "SeekCode(g='C', a=0, tr=False, tl=False, ert1=0, erh1=0, ert2=0, erh2=0, elt1=0, elh1=0, elt2=0, elh2=0, xr=False, xl=False, sr=False, sl=False, sb=False)"

REAL_CODES = [
    "B00T01E9___-____X00S___",
    "B60T11E90__-43__X00S001",
    "B90T11E8988-403_X00S000",
    "B60T11E008_-505_X00S00_",
    "B60T11E9_98-43__X00S00_",
    "B70T11E0090-405_X00S001",
    "B80T11E70__-30__X00S___",
    "B90T10E979_-34__X00S00_",
    "B70T11E80__-3_5_X0_S0_0",
    "B__T11E8___-_443X00S__1",
    "B90T11E9998-4334X00S___",
    "C00T00E0000-0000X00S000",
]

@pytest.mark.parametrize("code_str", REAL_CODES)
def test_round_trip(code_str):
    parsed = SeekCode.from_str(code_str)
    assert str(parsed) == code_str
    assert SeekCode.from_str(str(parsed)) == parsed


def test_single_digit_age_is_zero_padded_and_round_trips():
    code = SeekCode.from_str("B05T00E0000-0000X00S000")
    assert code.a == 5
    assert str(code) == "B05T00E0000-0000X00S000"


def test_seek_code_is_immutable():
    code = SeekCode.from_str("B00T00E0000-0000X00S000")
    with pytest.raises(dataclasses.FrozenInstanceError):
        code.g = "C"


def test_equality_and_hash():
    a = SeekCode.from_str("B60T11E90__-43__X00S001")
    b = SeekCode.from_str("B60T11E90__-43__X00S001")
    c = SeekCode.from_str("B60T11E90__-43__X00S000")
    assert a == b and hash(a) == hash(b)
    assert a != c
    assert {a, b, c} == {a, c}


INVALID_CODES = [
    "",
    "not a seek code",
    "X00T00E0000-0000X00S000",          # bad gender
    "B00T00E0000-0000X00S00",           # too short
    "B00T00E0000-0000X00S0000",         # too long
    "B00T22E0000-0000X00S000",          # bad tusk digit
    "B00T00E1000-0000X00S000",          # right ear sector "1" not allowed
    "B00T00E0000-7000X00S000",          # right-ear sector "7" on left side
    "B00T00E3000-0000X00S000",          # left-ear sector "3" on right side
    "B0AT00E0000-0000X00S000",          # non-digit age
    "B00X00E0000-0000X00S000",          # missing T marker
    "B00T00F0000-0000X00S000",          # missing E marker
    "B00T00E00000000X00S000",           # missing dash
]


@pytest.mark.parametrize("bad", INVALID_CODES)
def test_invalid_codes_raise(bad):
    with pytest.raises(ValueError):
        SeekCode.from_str(bad)


@pytest.mark.parametrize("kwargs", [
    {"g": "Z"},                        # bad gender
    {"a": 100},                        # age out of range
    {"a": -1},                         # age out of range
    {"a": True},                       # bool is not a valid age
    {"tr": 2},                         # bad boolean flag
    {"xl": "yes"},                     # bad boolean flag
    {"ert1": 3},                       # left sector on right ear
    {"elt1": 7},                       # right sector on left ear
    {"ert1": False},                   # bool is not a valid sector (False == 0)
    {"ert1": True},                    # bool is not a valid sector
    {"elt1": False},                   # bool is not a valid sector (False == 0)
    {"ert1": 7.0},                     # float is not a valid sector
    {"ert1": "7"},                     # str is not a valid sector
    {"a": 5.0},                        # float is not a valid age
])
def test_invalid_construction_raises(kwargs):
    base = dict(
        g="B", a=0, tr=False, tl=False,
        ert1=None, erh1=None, ert2=None, erh2=None,
        elt1=None, elh1=None, elt2=None, elh2=None,
        xr=None, xl=None, sr=None, sl=None, sb=None,
    )
    with pytest.raises(ValueError):
        SeekCode(**(base | kwargs))


def test_zero_sector_and_age_still_accepted():
    code = SeekCode(
        g="B", a=0, tr=False, tl=False,
        ert1=0, erh1=None, ert2=None, erh2=None,
        elt1=0, elh1=None, elt2=None, elh2=None,
        xr=None, xl=None, sr=None, sl=None, sb=None,
    )
    assert code.a == 0
    assert code.ert1 == 0 and code.elt1 == 0
    assert str(code) == "B00T00E0___-0___X__S___"
