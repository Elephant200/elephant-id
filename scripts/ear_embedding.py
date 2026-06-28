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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from elephant_id.coding.ears.tear_profile import TearProfile
from elephant_id.coding.photo_analyzer import PhotoAnalyzer
from elephant_id.dataset import Dataset
from elephant_id.log import configure_logging
from elephant_id.visualize import plot_tear_profile, plot_tear_profile_geometry

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

def main() -> None:
    load_dotenv()
    configure_logging()

    dataset = Dataset(
        dataset_root=REPO_ROOT / "dataset/elephants-alive/coded",
        metadata_path=REPO_ROOT / "dataset/elephants-alive/images.csv",
    )
    analyzer = PhotoAnalyzer(dataset=dataset)

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
            tear_profile: TearProfile = ear_data["tear_profile"]

            fig, (axi, axp) = plt.subplots(
                1,
                2,
                figsize=(16, 7),
                width_ratios=[1, 1.25],
            )
            plot_tear_profile_geometry(axi, image, ear_data["ear"], tear_profile)
            axi.legend(fontsize=9, loc="lower right")
            axi.set_title(f"{label} ({identifier})", fontsize=12)
            plot_tear_profile(axp, tear_profile, title="tear profile")
            fig.tight_layout()
            fig.savefig(out / f"{label}_{ear_side}.png", dpi=110)
            plt.close(fig)
    print(f"saved {out}/<label>_<side>.png for ({len(results)}) images)")


if __name__ == "__main__":
    main()
