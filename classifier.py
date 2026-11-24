# this file will contain the algorithm for classification  

from data_intake import load_train, load_test 
import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression

# load in the data 
train = load_train()
test = load_test()

train_ids = train.iloc[:,0]
test_ids = test.iloc[:,0]

X_train = train.iloc[:,1:11]
Y_train = train.iloc[:,11]
X_test = test.iloc[:,1:11]

print(X_train.shape, Y_train.shape, X_test.shape, test_ids.shape)
#print(Y_train.head())

# baseline algorithm 
def simple_logistic(X_train, Y_train, X_test):

    simple_logistic = LogisticRegression()
    simple_logistic.fit(X_train, Y_train)
    preds = simple_logistic.predict(X_test)

    return preds


# save results as csv file 
# make sure we have the correct output for kaggle 
logistic_preds = simple_logistic(X_train, Y_train, X_test)
logistic_result = np.column_stack((test_ids, logistic_preds))

np.savetxt("submission.csv", logistic_result, delimiter=",", fmt="%g", 
           header="ID,Y", comments = "")
