import numpy as np
import pandas as pd
import xgboost as xgb

print "loading..."

# read  data
dtrain=xgb.DMatrix(r"./data/filter_count_train_combine.txt")
dtest=xgb.DMatrix(r"./data/filter_count_final_combine.txt")
#train
print "training..."
param = {'n_estimators': 100,'max_depth':3, 'eta':0.1, 'silent':1,'min_child_weight': 5, 'objective':'binary:logistic','subsample':1,'colsample_bytree': 0.75,'eval_metric':'logloss','seed': 2017,'nthread':3}#,'gpu_id':1,'max_bin':16,'updater':'grow_gpu_hist'}
num_round=500
bst = xgb.train(param, dtrain, num_round)
#preds
print "predicting..."
preds = bst.predict(dtest)
print(preds.mean())
# submission
dfinstance=pd.read_csv("origin_data/test.csv")
df = pd.DataFrame({"instanceID": dfinstance["instanceID"].values, "proba": preds})
df.to_csv("submission.csv", index=False)
