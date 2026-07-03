"""Run the Alphaphant sidecar: ``uv run python -m elephant_id.api``."""

import argparse

import uvicorn
from dotenv import load_dotenv

from elephant_id.api.app import create_app
from elephant_id.log import configure_logging


def main() -> None:
    """Parse server options and serve the sidecar API."""
    load_dotenv()
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8756)
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
