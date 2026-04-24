import logging
import sys
from core.config import get_settings


def setup_logging() -> None:
    settings = get_settings()
    log_level = logging.DEBUG if settings.debug else logging.INFO

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Stop noisy third-party libraries from flooding your terminal
    logging.getLogger("pyannote").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)