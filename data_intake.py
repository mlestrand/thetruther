# this file will load in all of the data 

import pandas as pd

def load_train():
    return pd.read_csv("train.csv")
def load_test():
    return pd.read_csv("test.csv")

sample_submission = pd.read_csv("sample_submission.csv")
