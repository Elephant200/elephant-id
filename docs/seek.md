# SEEK Coding Specification

## Purpose
SEEK = **System for Elephant Ear-pattern Knowledge**. It is a fixed-format code used to identify individual African elephants.

In this project, SEEK is treated as a structured review output and interoperability format. The current field team does not rely heavily on manual SEEK coding, and there is no existing SEEK-code matching algorithm to plug into. The product should therefore assist reviewers in producing SEEK-compatible records rather than requiring users to manually code SEEK first.

Classic SEEK should be implemented as-is for v1. It is the first coding target and should stay faithful to the fixed-format grammar below.

The broader system should still leave room for later features. Future matching records may include additional structured fields, Curvrank contours and curvature signatures, reviewable curvature plots, learned vector embeddings, and other descriptors that help matching but do not fit into the original character grammar.

When implementing `SeekCode`, preserve the grammar in this document exactly. When implementing storage or matching, avoid assumptions that would make future non-SEEK descriptors impossible.

## Full code format
```
[Gender][Age]T[RightTusk][LeftTusk]E[R1][R2][R3][R4]-[L1][L2][L3][L4]X[RightExtreme][LeftExtreme]S[RightSpecial][LeftSpecial][BodySpecial]
```

Block order is fixed and contains no spaces.

In every position, `_` denotes **unknown** (age uses `__`, since the field is two characters wide).

## 1. Gender
One character.

- `B` = bull
- `C` = cow
- `_` = unknown

## 2. Age
Two characters representing birth-year bracket.

- `60` = 1900-1969
- `70` = 1970-1979
- `80` = 1980-1989
- `90` = 1990-1999
- `00` = 2000-2009
- `10` = 2010-2019
- `20` = 2020-2029
- `__` = unknown

## 3. Tusk block
Format: `T[RightTusk][LeftTusk]`

- `0` = absent
- `1` = present
- `_` = unknown

Broken tusks are not recorded, because they can regrow or break again.

## 4. Ear block
Format: `E[R1][R2][R3][R4]-[L1][L2][L3][L4]`

The right-ear block and left-ear block each contain **four ordered slots**. The slot meanings are fixed.

For each ear:

- slot `1` = position of the most prominent tear
- slot `2` = position of the most prominent hole
- slot `3` = position of the second most prominent tear
- slot `4` = position of the second most prominent hole

So:

- `R1` = most prominent tear on right ear
- `R2` = most prominent hole on right ear
- `R3` = second most prominent tear on right ear
- `R4` = second most prominent hole on right ear
- `L1` = most prominent tear on left ear
- `L2` = most prominent hole on left ear
- `L3` = second most prominent tear on left ear
- `L4` = second most prominent hole on left ear

The literal `-` between `R4` and `L1` is a fixed delimiter; it never appears inside a slot.

### Ear-feature definition
A **tear** includes a tear with or without a flap of skin hanging down.

### Ear position codes (per-side)
Positions are coded using the figure's clock-face mapping. The image is found on page seven of the SEEK paper. **Each side uses its own subset of position ids**:

- **Right ear** slots may only contain: `0`, `7`, `8`, `9`, or `_`
- **Left ear** slots may only contain: `0`, `3`, `4`, `5`, or `_`

Meanings:

- `3`, `4`, `5` = mapped left-ear positions from the figure
- `7`, `8`, `9` = mapped right-ear positions from the figure
- `0` = other (a feature exists at a position not covered by the figure)
- `_` = unknown / not recorded

A right-ear position id (`7/8/9`) appearing in a left-ear slot, or vice-versa, is a data-entry error and will be rejected by the parser.

### Ear coding rules
For each ear:
1. Find the most prominent tear -> slot 1
2. Find the most prominent hole -> slot 2
3. Find the second most prominent tear -> slot 3
4. Find the second most prominent hole -> slot 4

Tie-breaking:
- tears: deepest first
- holes: largest diameter first
- if still tied, code the feature that is higher on the ear according to the figure's position mapping

### Implementation notes
Curvrank-style contour features must be computed in a deterministic coordinate system. Ear contours should use stable side-aware orientation and deterministic start/end anchor points so the same ear produces comparable contour coordinates across runs.

Observed tear locations roughly around SEEK sectors 3 / 9, 4 / 8, and half of 5 / 7. This is an implementation hint for contour analysis only; valid left-ear position codes remain `0`, and `_` for both sides, `3`, `4`, `5` for left, and `9`, `8`, `7` for right.

## 5. Extreme block
Format: `X[RightExtreme][LeftExtreme]`

- `0` = absent
- `1` = present
- `_` = unknown

A tear or hole is classified as **extreme** when it:

- extends at least `1/4` of the way toward the inner ear in length, and/or
- is at least `1/4` of the total ear-margin width

Operationally, each ear gets one binary value indicating whether an extreme feature is present on that ear.

## 6. Special block
Format: `S[RightSpecial][LeftSpecial][BodySpecial]`

- `0` = absent
- `1` = present
- `_` = unknown

### Ear special features
These include:

- scars
- significant growths
- skin issues
- wavy ear
- floppy ear
- jagged ear
- any marking at the back of the top fold of the ear

### Body special features
These include:

- scars
- significant growths
- skin issues
- missing tail
- deformed tail

## Compact reference
```text
[Gender][Age]T[RT][LT]E[R1][R2][R3][R4]-[L1][L2][L3][L4]X[RX][LX]S[RS][LS][BS]

Length: 23 characters

Gender:
B = bull
C = cow
_ = unknown

Age (two chars):
60 = 1900-1969
70 = 1970-1979
80 = 1980-1989
90 = 1990-1999
00 = 2000-2009
10 = 2010-2019
20 = 2020-2029
__ = unknown

Tusk:
0 = absent
1 = present
_ = unknown
broken tusks not recorded

Ear slots per ear:
1 = most prominent tear
2 = most prominent hole
3 = second most prominent tear
4 = second most prominent hole

Ear position codes per-side:
right ear : {0, 7, 8, 9, _}
left ear  : {0, 3, 4, 5, _}
0 = other
_ = unknown

Ear tie-breaking:
- deepest tear first
- largest hole first
- if still tied, higher position first

Extreme:
0 = absent
1 = present
extreme = reaches >= 1/4 toward inner ear and/or >= 1/4 of total ear-margin width

Special:
RS = right-ear special feature present/absent
LS = left-ear special feature present/absent
BS = body special feature present/absent
```
