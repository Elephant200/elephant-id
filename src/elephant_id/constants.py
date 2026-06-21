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
# Tear profile (coding/ears/tear_profile.py) -- all lengths in units of S, the convex-
# hull arc length between the ear anchors
TEAR_PROFILE_BINS = 1024  # matches contour resampling
TEAR_TRIM_LO = 0.025      # zeroed leading fraction of x: reduces edge noise
TEAR_TRIM_HI = 0.025      # zeroed trailing fraction of x: reduces edge noise
TEAR_ALPHA_FRAC = 0.10    # rolling-disk radius / S (~20% of bbox long side)
TEAR_OPEN_FRAC = 0.007    # opening radius / S (~1.5% of bbox long side)
TEAR_SMOOTH_SIGMA = 2.0   # profile gaussian, bins

# Detection filtering thresholds
MIN_FEATURE_BODY_OVERLAP = 0.2
MIN_MULTIPLE_BODY_AREA_RATIO = 2
MIN_MULTIPLE_EAR_AREA_RATIO = 3
