import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sys
from src.exception import CustomException
def saveObj(path,data):
    try:
        with open(path, "wb") as file:
            pickle.dump(data, file)

    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)