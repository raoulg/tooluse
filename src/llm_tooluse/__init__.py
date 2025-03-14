import sys

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")
logger.add("logs/logfile.log", level="DEBUG")
