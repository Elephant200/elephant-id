# ==================== Cache configuration ====================
DEFAULT_CACHE_ROOT = ".cache"

# ==================== Roboflow API configuration ====================
ROBOFLOW_API_URL = "https://serverless.roboflow.com"
ROBOFLOW_WORKSPACE = "seek-identification"
ROBOFLOW_SAM3_WORKFLOW_ID = "sam3"

# ==================== Model configuration ====================
# SAM3 model configuration
SAM3_QUERY_PRESETS = {
    "features": (
        "elephant trunk",
        "tusk",
        "ear",
        "tail",
    ),
    "body": (
        "elephant",
    ),
}
DEFAULT_SAM3_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_SAM3_NMS = True
DEFAULT_SAM3_NMS_IOU_THRESHOLD = 0.2


# ==================== Algorithm configuration ====================
# Tear profile (coding/tears.py) -- all lengths in units of S, the convex-
# hull arc length between the ear anchors (rotation- and tear-invariant;
# ~2.07x the bounding-box long side on a typical ear). Chosen on a
# 17-photo / 8-individual pilot set; re-validate on held-out individuals.
TEAR_PROFILE_BINS = 1024  # 1 bin ~ 0.1% of S (1.5-5 px); smallest coded
                          # tears span ~10 bins; matches margin resampling
TEAR_TRIM_LO = 0.20       # zeroed leading fraction of x: outside SEEK coverage
TEAR_TRIM_HI = 0.10       # zeroed trailing fraction of x: outside SEEK coverage
TEAR_ALPHA_FRAC = 0.10    # rolling-disk radius / S (~20% of bbox long side)
TEAR_OPEN_FRAC = 0.007    # opening radius / S (~1.4% of bbox long side)
TEAR_SMOOTH_SIGMA = 2.0   # profile gaussian, bins (ablated: removing it
                          # costs separation; tangent smoothing did not and
                          # was removed)

# Detection filtering thresholds
MIN_FEATURE_BODY_OVERLAP = 0.2
MIN_MULTIPLE_BODY_AREA_RATIO = 2
MIN_MULTIPLE_EAR_AREA_RATIO = 3
