import asyncio
from pathlib import Path
from loguru import logger
from llm_tooluse import MCPToolLoader
import sys

logger.remove()
logger.add(sys.stdout, level="DEBUG")

async def main():
    loader = MCPToolLoader()
    # Assuming you have a server running at http://localhost:8000
    logger.info("\n=== Loading SSE server ===")

    try:
        toolcollection = await loader.load_server(
            name="sse_server",
            target="http://localhost:8000/mcp"
        )

        logger.info(f"Loaded {len(toolcollection)} tools via SSE")
        result = await toolcollection("add", a=5, b=3)
        logger.info(f"add(5, 3) = {result}")

        # Use the SSE server
        # result = await ml_collection("some_tool", arg="value")

    except Exception as e:
        logger.warning(f"SSE server not available: {e}")
        logger.info("To test SSE, start a server with: python ml_server.py")

    # Cleanup
    await loader.cleanup()
    logger.info("\n✅ Done!")

if __name__ == "__main__":
    logger.info("Starting SSE test...")
    asyncio.run(main())
