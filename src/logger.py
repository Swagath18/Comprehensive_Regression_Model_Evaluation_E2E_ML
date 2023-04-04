#logging all information for us to track all the errors and log into text file
import logging 
import os
from datetime import datetime

#format in which we want to log
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

#providing path for the log file, getcwd: getting current workind directory
logs_path=os.path.join(os.getcwd(),"logs", LOG_FILE)

#creating directory and exist ok is to keep appending to the same file
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH=os.path.join(logs_path, LOG_FILE)

#inorder to overwrite the logging we have to set it to basic config
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO, # printing the messages in info type

)

'''
###############____Testing purpose___________#######
if __name__=="__main__":
    logging.info("Logging has started")
#Just to give you some idea on how the logger information would look like
#[2023-04-03 21:10:06,274] 26 root - INFO - Logging has started
#above shows logger info, 26 root means 26th line print message, creates log file in menstioned formate.log
##########################################################
'''