# this file will load in all of the data 

import pandas as pd

def load_train():
    return pd.read_csv("data/train.csv")
def load_test():
    return pd.read_csv("data/test.csv")

sample_submission = pd.read_csv("data/sample_submission.csv")
