from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/logfile.log", level="DEBUG")