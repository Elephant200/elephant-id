"""Visual QA for the ear-margin embedding: photo next to its profile.

For each manifest photo: the ear image with the margin (white), the alpha
reference (cyan), and the deepest scan ray (red) overlaid -- beside the 1-D
profile, with detected tear events marked. This is how embedding accuracy is
checked against the actual image: every bump in the profile should point at
a visible tear, and the profile should read as the ear unrolled flat.

Outputs (outputs/ear_embedding_v2/): <label>.png per photo.

Run:  uv run python -m scripts.ear_embedding
"""
from pathlib import Path
from typing import Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

from elephant_id.coding.ears.tear_profile import TearProfile, polar_directions
from elephant_id.coding.photo_analyzer import PhotoAnalyzer
from elephant_id.constants import TEAR_PROFILE_BINS, TEAR_TRIM_DEGREES
from elephant_id.dataset import Dataset
from elephant_id.image.transforms import apply_crop
from elephant_id.log import configure_logging

REPO_ROOT = Path(__file__).resolve().parent.parent

PHOTOS = {
    "adam1": "Adam_2011-03-31_03",
    "adam2": "Adam_2011-03-31_07",
    "ripley1": "Ripley_2008-06-25_06",
    "ripley2": "Ripley_2016-04-19_13",
    "les1": "Les_2007-05-03_08",
    "larson1": "Larson_2018-02-13_09",
    "delani1": "Delani_2008-12-16_01",
    "nile1": "Nile_2017-06-21_03",
    "nile2": "Nile_2014-08-21_39",
    "nile3": "Nile_2017-09-30_06",
    "nile4": "Nile_2016-08-24_27",
    "snap1": "Snap_2008-10-17_07",
    "snap2": "Snap_2008-10-17_06",
    "snap3": "Snap_2007-06-10_08",
    "snap4": "Snap_2007-06-10_22",
}


def draw_polar_guides(
    axis: plt.Axes,
    original_anchor_points: tuple[tuple[float, float], tuple[float, float]],
    side: Literal["left", "right"],
    radius: float,
    spoke_length: float,
    offset: np.ndarray,
) -> None:
    """Draw the polar calibration semicircle, angle rays, and trim boundaries."""
    upper_anchor = np.asarray(original_anchor_points[0], dtype=float)
    lower_anchor = np.asarray(original_anchor_points[1], dtype=float)
    arc_angles = np.linspace(0.0, 180.0, 181)
    midpoint, arc_directions = polar_directions(
        upper_anchor,
        lower_anchor,
        side,
        arc_angles,
    )
    arc = midpoint + radius * arc_directions
    axis.plot(*(arc - offset).T, color="gold", linewidth=0.9, alpha=0.8)

    guide_angles = np.arange(0.0, 181.0, 30.0)
    _, guide_directions = polar_directions(
        upper_anchor,
        lower_anchor,
        side,
        guide_angles,
    )
    for direction in guide_directions:
        axis.plot(
            [midpoint[0] - offset[0], midpoint[0] + spoke_length * direction[0] - offset[0]],
            [midpoint[1] - offset[1], midpoint[1] + spoke_length * direction[1] - offset[1]],
            color="gold",
            linewidth=0.7,
            alpha=0.65,
        )

    _, trim_directions = polar_directions(
        upper_anchor,
        lower_anchor,
        side,
        np.array((TEAR_TRIM_DEGREES, 180.0 - TEAR_TRIM_DEGREES)),
    )
    for direction in trim_directions:
        axis.plot(
            [midpoint[0] - offset[0], midpoint[0] + radius * direction[0] - offset[0]],
            [midpoint[1] - offset[1], midpoint[1] + radius * direction[1] - offset[1]],
            color="tab:orange",
            linestyle="--",
            linewidth=1.0,
        )


def main() -> None:
    load_dotenv()
    configure_logging()

    dataset = Dataset(
        dataset_root=REPO_ROOT / "dataset/elephants-alive/coded",
        metadata_path=REPO_ROOT / "dataset/elephants-alive/images.csv",
    )
    analyzer = PhotoAnalyzer(dataset=dataset)
    angles_degrees = np.linspace(0.0, 180.0, TEAR_PROFILE_BINS)
    coded_angle_mask = (
        (angles_degrees > TEAR_TRIM_DEGREES)
        & (angles_degrees < 180.0 - TEAR_TRIM_DEGREES)
    )

    out = REPO_ROOT / "outputs" / "ear_embedding_v2"
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for label, identifier in PHOTOS.items():
        photo = dataset.get_photo(identifier)
        analysis = analyzer.analyze(photo)
        ear_data: list[dict] = analysis["ears"]
        results[label] = {ear["ear"].side: ear for ear in ear_data}

    # per-photo: ear image with overlays and profile with events
    for label, identifier in PHOTOS.items():
        if label not in results:
            continue
        image = dataset.read_image(dataset.get_photo(identifier))

        for ear_side, ear_data in results[label].items():
            ear_image = apply_crop(image, ear_data["ear"].xyxy)
            offset = np.array([ear_data["ear"].xyxy[0], ear_data["ear"].xyxy[1]])

            contour = ear_data["ear"].resampled_contour(1024)
            tear_profile: TearProfile = ear_data["tear_profile"]

            k = int(np.argmax(np.where(coded_angle_mask, tear_profile.profile, -np.inf)))

            fig, (axi, axp) = plt.subplots(
                1, 2, figsize=(16, 7), width_ratios=[1, 1.25])
            axi.imshow(cv2.cvtColor(ear_image, cv2.COLOR_BGR2RGB))
            axi.plot(*(contour - offset).transpose(), "w", lw=1.1, alpha=0.9, label="contour")
            axi.plot(*(tear_profile.reference - offset).transpose(), "tab:cyan", lw=1.5,
                    label="alpha reference")
            draw_polar_guides(
                axi,
                ear_data["ear"].original_anchor_points,
                ear_side,
                tear_profile.scale,
                float(np.hypot(*ear_image.shape[:2])),
                offset,
            )
            hit = tear_profile.origins[k] + tear_profile.profile[k] * tear_profile.scale * tear_profile.normals[k]
            axi.plot([tear_profile.origins[k, 0] - offset[0], hit[0] - offset[0]],
                    [tear_profile.origins[k, 1] - offset[1], hit[1] - offset[1]],
                    "r-", lw=2.0, label="deepest tear")
            axi.legend(fontsize=9, loc="lower right")
            axi.set_title(f"{label} ({identifier})", fontsize=12)
            height, width = ear_image.shape[:2]
            axi.set(xlim=(-0.5, width - 0.5), ylim=(height - 0.5, -0.5))
            axi.axis("off")

            axp.plot(angles_degrees, tear_profile.profile, "tab:red",
                    lw=1.4)
            axp.axvspan(0, TEAR_TRIM_DEGREES, color="0.85")
            axp.axvspan(180 - TEAR_TRIM_DEGREES, 180, color="0.85")
            axp.axvline(TEAR_TRIM_DEGREES, color="tab:orange", linestyle="--", lw=1.0)
            axp.axvline(180 - TEAR_TRIM_DEGREES, color="tab:orange", linestyle="--", lw=1.0)
            axp.set_xlim(0, 183)
            axp.set_xticks((0, 30, 60, 90, 120, 150, 180))
            axp.set_ylim(-0.03, 0.4)
            axp.set_xlabel("ear angle (degrees)")
            axp.set_ylabel("tear depth / R")
            axp.set_title("tear profile", fontsize=12)
            axp.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out / f"{label}_{ear_side}.png", dpi=110)
            plt.close(fig)
    print(f"saved {out}/<label>_<side>.png for ({len(results)}) images)")


if __name__ == "__main__":
    main()
