import os; os.chdir('')
import logger
log = logger.setup_applevel_logger()

import pandas as pd
import file_2

from datetime import datetime
dttm = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

log.info("this is from file 1")

print("Hello World from File 1")
file2_data = file_2.load_data()
print(file2_data)
