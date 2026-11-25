# this file will contain the algorithm for classification  

from data_intake import load_train, load_test 
import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# load in the data 
train = load_train()
test = load_test()

train_ids = train.iloc[:,0]
test_ids = test.iloc[:,0]

X_train = train.iloc[:,1:11]
Y_train = train.iloc[:,11]
X_test = test.iloc[:,1:11]

p = np.mean(Y_train)
print(p)

#print(X_train.shape, Y_train.shape, X_test.shape, test_ids.shape)
#print(X_train.head())

# baseline algorithm 
def simple_logistic(X_train, Y_train, X_test):

    simple_logistic = LogisticRegression(class_weight='balanced') # balance bc positives are a lot less prevalent
    simple_logistic.fit(X_train, Y_train)
    preds = simple_logistic.predict(X_test)

    return preds

# save results as csv file 
# make sure we have the correct output for kaggle 
logistic_preds = simple_logistic(X_train, Y_train, X_test)
logistic_result = np.column_stack((test_ids, logistic_preds))

np.savetxt("submission.csv", logistic_result, delimiter=",", fmt="%g", 
           header="ID,Y", comments = "")

print(np.mean(logistic_preds))

# checking potential score using out of fold cv with claude: 
# model = make_pipeline(StandardScaler(),
#                       LogisticRegression(max_iter=1000, class_weight="balanced"))
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# oof_probs = cross_val_predict(model, X_train, Y_train, cv=cv,
#                               method="predict_proba")[:, 1]
# print("OOF AUC:", roc_auc_score(Y_train, oof_probs))
# print("F1 at 0.5:", f1_score(Y_train, oof_probs >= 0.5))
