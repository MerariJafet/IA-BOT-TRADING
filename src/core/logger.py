import logging
import sys
from typing import Optional

_DEFAULT_FORMAT = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"


def get_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    fmt: str = _DEFAULT_FORMAT,
) -> logging.Logger:
    """Create or get a configured logger.

    Args:
        name: Logger name (module name). None for root.
        level: Logging level (e.g., logging.INFO).
        fmt: Log line format.

    Returns:
        logging.Logger: configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
