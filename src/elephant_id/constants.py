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
# Tear profile (coding/ears/tear_profile.py) -- lengths are fractions of R, the
# equal-area semicircle radius of the anchored ear mask.
TEAR_PROFILE_BINS = 720
TEAR_TRIM_DEGREES = 5.0
TEAR_ALPHA_FRAC = 0.35
TEAR_OPEN_FRAC = 0.020
TEAR_SMOOTH_SIGMA = 2.0

# Detection filtering thresholds
MIN_FEATURE_BODY_OVERLAP = 0.2
MIN_MULTIPLE_BODY_AREA_RATIO = 2
MIN_MULTIPLE_EAR_AREA_RATIO = 3
