import logger
log = logger.get_logger(__name__)

import pandas as pd

from datetime import datetime
dttm = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

log.info("this is from file 2")

def load_data():
    return 'this is return value from file2.load_data()'
