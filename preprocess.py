# This Python file uses the following encoding: utf-8

import pandas as pd


print("Loading file ...")

# 用户基础特征
user = pd.read_csv("origin_data/user.csv")
# App特征
app = pd.read_csv("origin_data/app_categories.csv")
# 广告特征
ad = pd.read_csv("origin_data/ad.csv")
# 广告位特征
position = pd.read_csv("origin_data/position.csv")

train = pd.read_csv("origin_data/train.csv")
test = pd.read_csv("origin_data/test.csv")

train = train.drop(['conversionTime'], axis=1)
# train = train.drop(['clickTime'], axis=1)
# test = test.drop(['clickTime'], axis=1)

print("Done")
print("Construct train set ...")

print("Merge user ...")
train = pd.merge(train, user, on='userID')
test = pd.merge(test, user, on='userID')

print("Merge ad ...")
train = pd.merge(train, ad, on='creativeID')
test = pd.merge(test, ad, on='creativeID')

print("Merge app ...")
train = pd.merge(train, app, on='appID')
test = pd.merge(test, app, on='appID')

print("Merge position ...")
train = pd.merge(train, position, on='positionID')
test = pd.merge(test, position, on='positionID')

# train = train.dropna(axis=0, how='any')

print("Output ...")
train.to_csv("output/train_process.csv", index=False)
test.to_csv("output/test_process.csv", index=False)
