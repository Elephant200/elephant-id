import numpy as np

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
# Detection filtering thresholds
MIN_FEATURE_BODY_OVERLAP = 0.2
MIN_MULTIPLE_BODY_AREA_RATIO = 2
MIN_MULTIPLE_EAR_AREA_RATIO = 3

# Curvrank defaults
DEFAULT_CURVATURE_RADII = np.array([0.02, 0.04, 0.06, 0.08, 0.10], dtype=np.float32)
DEFAULT_CURVATURE_WEIGHTS = np.array([0.6, 0.9, 1.0, 0.9, 0.6], dtype=np.float32)
