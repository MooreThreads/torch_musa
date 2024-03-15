"""logger util"""
import logging.config
import sys
from logging import Formatter, StreamHandler


def create_logger():
    """create logger instance"""
    log_format = "%(asctime)s | %(module)s | %(thread)d | %(levelname)s : %(message)s"
    log_date_fmt = "%Y-%m-%d %H:%M:%S"

    # set logger
    logging.captureWarnings(True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = Formatter(fmt=log_format, datefmt=log_date_fmt)

    console_handler = StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(fmt)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    return logger


LOGGER = create_logger()

if __name__ == '__main__':
    LOGGER.debug('debug')
    LOGGER.info('info')
    LOGGER.warning('warn')
    LOGGER.error('error')
    LOGGER.critical('critical')
