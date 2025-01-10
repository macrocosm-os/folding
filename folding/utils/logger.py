import sys
from loguru import logger

# Remove any existing handlers
logger.remove()

# Custom format for different log levels
FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level}</level> | <level>{message}</level>"

# Add default stdout handler with the format
logger.add(
    sys.stdout,
    format=FORMAT,
    level="TRACE",
    enqueue=True,
    colorize=False,
    catch=True,
    backtrace=False,
    diagnose=False,
)


def setup_file_logging(log_path, retention):
    """Setup file logging with rotation"""
    logger.add(
        log_path,
        rotation=retention,
        serialize=False,
        enqueue=True,
        backtrace=False,
        diagnose=False,
        level="TRACE",
        format=FORMAT,
    )


def add_events_level():
    """Add custom EVENTS level if it doesn't exist"""
    if "EVENTS" not in logger._core.levels:
        logger.level("EVENTS", no=38, icon="📝")


__all__ = ["logger", "setup_file_logging", "add_events_level"]
