import os #Used to interact with the operating system
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #This line is a fix for PyTorch errors
import sys # used for system level operations
 
# importing project modeules 
from xray.exception import XRayException
from xray.pipeline.train_pipeline import TrainPipeline #this imports your main ML pipeline controller

def start_training():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

    except Exception as e:
        raise XRayException(e, sys)


if __name__ == "__main__":
    start_training()
