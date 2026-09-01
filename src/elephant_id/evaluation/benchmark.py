"""Loading of ID-only retrieval benchmark manifests."""

import csv
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

_COLUMNS = (
    "known_elephant_name",
    "sighting_id",
    "left_photo_id",
    "right_photo_id",
)


@dataclass
class RetrievalBenchmark:
    """Benchmark declarations grouped by known elephant."""

    sightings: dict[str, dict[UUID, tuple[UUID, UUID]]]


class BenchmarkValidationError(ValueError):
    """Report every validation error found before matching begins."""

    def __init__(self, errors: tuple[str, ...]) -> None:
        """Initialize the error with all discovered problems."""
        self.errors = errors
        super().__init__("Invalid retrieval benchmark:\n" + "\n".join(errors))


def _parse_uuid4(
    value: str | None,
    field: str,
    row_number: int,
    errors: list[str],
) -> UUID | None:
    """Parse one canonical UUIDv4 cell or record its error."""
    if not value:
        errors.append(f"Row {row_number}: missing {field}")
        return None
    try:
        parsed = UUID(value)
    except ValueError:
        errors.append(f"Row {row_number}: invalid {field}: {value!r}")
        return None
    if parsed.version != 4 or str(parsed) != value:
        errors.append(f"Row {row_number}: {field} must be a canonical UUIDv4")
        return None
    return parsed


def load_benchmark(path: Path) -> RetrievalBenchmark:
    """Parse and validate a retrieval benchmark manifest.

    Raises:
        BenchmarkValidationError: If the manifest contains any invalid rows.
    """
    declarations: dict[str, dict[UUID, tuple[UUID, UUID]]] = {}
    seen_sightings: set[UUID] = set()
    errors: list[str] = []

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = tuple(reader.fieldnames or ())
        if columns != _COLUMNS:
            raise BenchmarkValidationError(
                (f"Manifest columns must be {_COLUMNS}, got {columns}",)
            )

        for row_number, row in enumerate(reader, start=2):
            row_error_count = len(errors)
            extra_values = row.get(None)
            if extra_values:
                errors.append(
                    f"Row {row_number}: expected {len(_COLUMNS)} values, "
                    f"got {len(_COLUMNS) + len(extra_values)}"
                )
            name = row["known_elephant_name"]
            if not name or not name.strip():
                errors.append(f"Row {row_number}: missing known_elephant_name")
            sighting_id = _parse_uuid4(
                row["sighting_id"], "sighting_id", row_number, errors
            )
            left_photo_id = _parse_uuid4(
                row["left_photo_id"], "left_photo_id", row_number, errors
            )
            right_photo_id = _parse_uuid4(
                row["right_photo_id"], "right_photo_id", row_number, errors
            )
            if sighting_id is not None:
                if sighting_id in seen_sightings:
                    errors.append(f"Row {row_number}: duplicate sighting_id {sighting_id}")
                seen_sightings.add(sighting_id)
            if len(errors) == row_error_count:
                assert sighting_id is not None
                assert left_photo_id is not None
                assert right_photo_id is not None
                declarations.setdefault(name, {})[sighting_id] = (
                    left_photo_id,
                    right_photo_id,
                )

    if not declarations and not errors:
        errors.append("Manifest contains no benchmark rows")
    if errors:
        raise BenchmarkValidationError(tuple(errors))
    return RetrievalBenchmark(declarations)
