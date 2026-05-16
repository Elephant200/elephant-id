import pytest

from elephant_id.domain import SeekCode


def test_seek_code_from_str_all_zero():
    code = SeekCode.from_str("B00T00E0000-0000X00S000")
    assert code.g == "B"
    assert code.a == 0
    assert code.tr is False and code.tl is False
    assert code.rt1 == 0 and code.rh1 == 0 and code.rt2 == 0 and code.rh2 == 0
    assert code.lt1 == 0 and code.lh1 == 0 and code.lt2 == 0 and code.lh2 == 0
    assert code.xr is False and code.xl is False
    assert code.sr is False and code.sl is False and code.sb is False


def test_seek_code_from_str_all_unknown():
    code = SeekCode.from_str("___T__E____-____X__S___")
    for v in (code.g, code.a, code.tr, code.tl,
              code.rt1, code.rh1, code.rt2, code.rh2,
              code.lt1, code.lh1, code.lt2, code.lh2,
              code.xr, code.xl, code.sr, code.sl, code.sb):
        assert v is None


def test_seek_code_to_str():
    code = SeekCode(
        g="C", a=0, tr=False, tl=False,
        rt1=0, rh1=0, rt2=0, rh2=0, lt1=0, lh1=0, lt2=0, lh2=0,
        xr=False, xl=False, sr=False, sl=False, sb=False,
    )
    assert str(code) == "C00T00E0000-0000X00S000"

REAL_CODES = [
    "B00T01E9___-____X00S___",
    "B60T11E90__-43__X00S001",
    "B90T11E8988-403_X00S000",
    "B60T11E008_-505_X00S00_",
    "B70T11E0090-405_X00S001",
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
    {"rt1": 3},                        # left sector on right ear
    {"lt1": 7},                        # right sector on left ear
])
def test_invalid_construction_raises(kwargs):
    base = dict(g="B", a=0, tr=False, tl=False)
    with pytest.raises(ValueError):
        SeekCode(**(base | kwargs))
