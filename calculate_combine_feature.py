import numpy as np
import pandas as pd

#read data, encode click time,split data
def calc_week_time(x):
    day=(int(x/10000)-17)%7
    hour=int(x%10000/100/3)
    #minute_5=int(x%100/5)
    #return day*24*12+hour*12+minute_5
    return day  


#univariate
def univariate_calculate_clickorconvert(x,name,querylist):
    if(x[name] in querylist): 
        return querylist[x[name]]
    else:
        return 0

def univariate_calculate_cvr(x,name):
    if(x[name+'_click']!=0):
        return x[name+'_convert']/x[name+'_click']
    else:
        return np.nan
    
def create_univariate_count(name):
    labelall=data_history[name].value_counts()
    labeltrue=data_history[data_history.label==1][name].value_counts()
    data_train_test[name+'_click']=data_train_test.apply(univariate_calculate_clickorconvert,name=name,querylist=labelall,axis=1) 
    print(name+' click finish')
    data_train_test[name+'_convert']=data_train_test.apply(univariate_calculate_clickorconvert,name=name,querylist=labeltrue,axis=1) 
    print(name+' convert finish')
    data_train_test[name+'_cvr']=data_train_test.apply(univariate_calculate_cvr,name=name,axis=1) 
    print(name+' all finish')
    return

#multivariate
def multivariate_calculate_click(x):
    try:
        return labelall.ix[x[name1]].ix[x[name2]]
    except:
        return 0

def multivariate_calculate_convert(x):
    try:
        return labeltrue.ix[x[name1]].ix[x[name2]]
    except:
        return 0

def multivariate_calculate_cvr(x):
    try:
        return x[name1+name2+'_convert']/x[name1+name2+'_click']
    except:
        return np.nan
    
def create_multivariate_count(fname1,fname2):
    global name1
    global name2
    global labelall
    global labeltrue
    name1=fname1  
    name2=fname2
    labelall=data_history.groupby([name1,name2]).size()
    labeltrue=data_history[data_history.label==1].groupby([name1,name2]).size()     
    data_train_test[name1+name2+'_click']=data_train_test.apply(multivariate_calculate_click,axis=1) 
    print(name1+'_'+name2+' click finish')
    data_train_test[name1+name2+'_convert']=data_train_test.apply(multivariate_calculate_convert,axis=1) 
    print(name1+'_'+name2+' convert finish')
    data_train_test[name1+name2+'_cvr']=data_train_test.apply(multivariate_calculate_cvr,axis=1) 
    print(name1+'_'+name2+' all finish')
    return

def main():
    #calculate univarite
    create_univariate_count('week_time')
    #calculate weektime and other feature
    calc_list=[ 'creativeID', 'userID','positionID', 'connectionType', 'telecomsOperator', 'adID',
       'camgaignID', 'advertiserID', 'appID', 'appPlatform', 'age', 'gender',
       'education', 'marriageStatus', 'haveBaby', 'hometown', 'residence',
       'sitesetID', 'positionType', 'appCategory', 'appinstalledcounts']
    for i in calc_list:
        create_multivariate_count('week_time',i)
    print('all clear')
    return

data_all=pd.read_csv(r"F:\txadcompetition\data\pre\myfeature\pure-merge.csv")
data_all['week_time']=data_all['clickTime'].apply(calc_week_time)
data_history=data_all[data_all.clickTime<240000]
data_train_test=data_all[data_all.clickTime>=240000]
main()
data_train_test.to_csv(r"F:\txadcompetition\data\pre\myfeature\data148-0601.csv",index=None)
