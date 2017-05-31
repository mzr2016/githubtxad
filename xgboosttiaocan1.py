import numpy as np
import pandas as pd
import xgboost as xgb

from loss_function import log_loss
print "loading..."

# read  data
#dtrain=xgb.DMatrix(r"./data/filter_count_train_combine.txt")
#dtest=xgb.DMatrix(r"./data/filter_count_final_combine.txt")
#train
dtrain = xgb.DMatrix(r"./data/filter_count_train_combine.txt")

sz = dtrain.num_row()
train = dtrain.slice([i for i in range(int(sz * 0.7))])
test = dtrain.slice([i for i in range(int(sz * 0.7),sz)])

#print sz, train.num_row() , test.num_row()
train_Y = train.get_label()

test_Y = test.get_label()
#if None in test_Y:
#    print "yes"
#else: print "no"
#print len(test_Y)

print "training..."
for i in range(100,800,50):
    #for j in np.arange(0.1,1,0.1):
        param = {'n_estimators': i,
                 'max_depth':5, 
                 'eta':0.01, 
                 'silent':1,
                 'min_child_weight': 5, 
                 'objective':'binary:logistic',
                 'subsample':1,
                 'colsample_bytree': 0.75,
                 'eval_metric':'logloss',
                 'seed': 2017,
                 'nthread':-1, 
                 'early_stopping_rounds':20, 
                 'gpu_id':1,'max_bin':16,'updater':'grow_gpu_hist'}
        num_round=1000

        #bst = xgb.train(param, train, num_round)
#preds
        #print "predicting..."
        #preds = bst.predict(test)
        #print("n_estimators",i,"means=",preds.mean(),"logloss=",log_loss(test_Y, preds))
# submission
#dfinstance=pd.read_csv("%s/test.csv"%data_root)
#df = pd.DataFrame({"instanceID": dfinstance["instanceID"].values, "proba": preds})
#df.to_csv("F:/txadcompetition/data/ans/submission.csv", index=False)
