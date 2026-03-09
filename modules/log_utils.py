import logging
from pathlib import Path

LOGGER_NAME = "sisifos"
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"


def setup_logger(log_file: str | Path = "logs/run2.txt", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(_DEFAULT_FORMAT)

    stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
    if not stream_handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    else:
        for handler in stream_handlers:
            handler.setLevel(level)
            handler.setFormatter(formatter)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        existing_file_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path.resolve()
        ]
        if not existing_file_handlers:
            file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        else:
            for handler in existing_file_handlers:
                handler.setLevel(level)
                handler.setFormatter(formatter)

    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)

