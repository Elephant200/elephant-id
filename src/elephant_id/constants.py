
# Model configuration
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

# Cache configuration
DEFAULT_CACHE_ROOT = ".cache"

# Roboflow configuration
ROBOFLOW_API_URL = "https://serverless.roboflow.com"
ROBOFLOW_WORKSPACE = "seek-identification"
ROBOFLOW_SAM3_WORKFLOW_ID = "sam3"

# Seek identification thresholds
MIN_FEATURE_BODY_OVERLAP = 0.2
