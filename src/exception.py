import sys #used to get information about the exception that occurred

# this is imported to check whether everthing is working fine, when you are writing for the first time do not import
#import logging 

#custom exception handling
def error_message_detail(error, error_detail:sys): #error_detail will be present inside sys

    _,_,exc_tb = error_detail.exc_info() #method returns a tuple of three items:type of exception, the exception instance itself, and traceback object
    file_name=exc_tb.tb_frame.f_code.co_filename 
    error_message="Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno,str(error))

    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):#contructor
        super().__init__(error_message) #inheriting from super class Exception
        self.error_message=error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):# print the error message
        return self.error_message
'''
################_testing purpose we are testing with random example_DONOT if you are not testing_#####
if __name__=="__main__":
    try:
        a=1/0
    except Exception as e:
        logging.info("Divide by Zero")
        raise CustomException(e, sys)
#############################################################
'''
