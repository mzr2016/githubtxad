import pandas as pd
import numpy as np
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn.externals import joblib
from loss_function import log_loss
import zipfile
#from sklearn.cross_validation import train_test_split
import datetime
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat
from operator import itemgetter
import random
import os
import time
import glob

X_train = pd.read_csv("output/X_train.csv")
X_test = pd.read_csv("output/X_test.csv")
y_train = X_train['label']
y_test = X_test['label']


    


predictors = ['clickTime', 'creativeID', 'positionID', 'adID', 'camgaignID', 'advertiserID', 'appID']

X_train = X_train[predictors]
X_test = X_test[predictors]


xgb = XGBClassifier(
     learning_rate=0.1,
     n_estimators=1000,
     max_depth=5,
     min_child_weight=5,
     gamma=0,
     subsample=0.8,
     colsample_bytree=0.8,
     objective='binary:logistic',
     scale_pos_weight=1,
     seed=0,
     silent=1
	)
eval_set = [(X_test, y_test)]  
#xgb.fit(X_train, y_train,verbose=True, early_stopping_rounds=50, eval_metric="logloss", eval_set=eval_set)



#gbdt = GradientBoostingClassifier(
#      loss='deviance'
#)
depth_range = np.arange(2, 5, 1)
rate_range = np.arange(0.1, 1.1, 0.1)
param_grid = dict(max_depth=depth_range, learning_rate=rate_range)
cv = StratifiedShuffleSplit(n_splits=5, random_state=25)
grid = GridSearchCV(xgb, param_grid=param_grid, cv=cv)
grid.fit(X_train, y_train)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

#proba_test = xgb.predict_proba(X_test)[:,1]

#print("computing loss function...")
#print(log_loss(y_test, proba_test))
