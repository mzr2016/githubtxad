import pandas as pd
import numpy as np
from scipy.stats import mode 

train = pd.read_csv("output/train_process.csv")
#test = pd.read_csv("output/test_positionrate.csv")



#train["userapp"] = train.groupby("age").transform(lambda x: x.fillna(x.mean()))["userapp"]

#test["userapp"] = test.groupby("age").transform(lambda x: x.fillna(x.mean()))["userapp"]




X_train = train[train['clickTime']<=285000]

X_test = X_train[X_train['clickTime']>275000]

X_train = X_train[X_train['clickTime']<=275000]

#a = X_test.ix[541734]
#b = X_test.ix[594054]

#c= pd.concat([a,b],axis=1)

#X_train = pd.concat([X_train,c],axis=1)



print("Output ...")
X_train.to_csv("output/X_train.csv", index=False)
X_test.to_csv("output/X_test.csv", index=False)
#test = test.to_csv("output/test_process_new.csv")
