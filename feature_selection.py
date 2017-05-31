import pandas as pd
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("output/train_simple_feature.csv")

data = data.dropna()

label = data['label']
train = data.drop(['label'], axis=1)

#gbrt = GradientBoostingRegressor()
#gbrt.fit(train, label)
#print(gbrt.feature_importances_)

lr = LogisticRegression(penalty='l2')
lr.fit(train, label)
print(lr.coef_)
                            #lr                       gbrt
# 'clickTime'           0.06111967              0.05260315          
# 'creativeID'          0.08602353  +           0.06635283  +
# 'userID'              0.03709443              0.0018216 
# 'positionID'          0.15735575  +           0.14642828  +
# 'connectionType'      0.03310876              0.05416907
# 'telecomsOperator'    0.00425786              0
# 'age'                 0.05655816              0.0242106
# 'gender'              0.01798962              0.01779665
# 'education'           0.01111859              0.00050296
# 'marriageStatus'      0.00234406              0
# 'haveBaby'            0.00744142              0
# 'hometown'            0.01749523              0.01329039
# 'residence'           0.01868252              0.00429357
# 'adID'                0.0547509   +           0.07259995  +
# 'camgaignID'          0.05881866  +           0.09887888  +
# 'advertiserID'        0.09483477  +           0.14314359  +
# 'appID'               0.08791078  +           0.10237394  +
# 'appPlatform'         0.00180424              0.00129512
# 'appCategory'         0.11513752  +           0.13130216  +
# 'sitesetID'           0.04093563              0.03548178
# 'positionType'        0.03521791              0.0334555 
