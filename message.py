# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging

def init_logger():
    log_file = '../logs/etl.logs'
    log_file_level=logging.NOTSET
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_file_level)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger
