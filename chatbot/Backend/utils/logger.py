import logging
import sys


def get_logger(name: str, config) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    level = logging.INFO
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
