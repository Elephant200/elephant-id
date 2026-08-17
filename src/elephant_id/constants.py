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
