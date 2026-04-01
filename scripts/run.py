"""Quick-start script for running the multi-agent system."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mas.pipeline import MASPipeline


async def main():
    pipeline = MASPipeline()

    # Example: process a document
    result = await pipeline.run(
        query="Extract key information from the provided document",
        context={"file_path": "data/sample.pdf"},
    )

    import json
    print(json.dumps(result, indent=2, default=str))

    # Print cost summary
    print("\n--- Cost Summary ---")
    print(json.dumps(pipeline.cost_tracker.get_summary(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
