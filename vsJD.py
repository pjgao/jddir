#coding:utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
pd.__version__

#读取登录数据
loginData=pd.read_csv('t_login.csv',dtype={'log_id':str,'id':str,'device':str,'log_from':str,'ip':str,'city':str,'result':str,'type':str})
loginDataTest=pd.read_csv('t_login_test.csv',dtype={'log_id':str,'id':str,'device':str,'log_from':str,'ip':str,'city':str,'result':str,'type':str})
loginData=loginData.append(loginDataTest)
loginData['time']=pd.to_datetime(loginData['time'])
######重排索引，不然后续的根据索引来连接会出错####
loginData=loginData.reset_index(drop=True)
del loginDataTest

#读取交易数据
tradeData=pd.read_csv("trade_2_login.csv",dtype={'id':str,'login_id':str})
tradeData.rename(columns={'login_id':'log_id'},inplace=True)
tradeData['time']=pd.to_datetime(tradeData['time'])

tradeTestData=pd.read_csv("t_trade_test_2_login.csv",dtype={'id':str,'login_id':str})
tradeTestData.rename(columns={'login_id':'log_id'},inplace=True)
tradeTestData['time']=pd.to_datetime(tradeTestData['time'])

#得到用户id，并以id为index用于后续好添加用户相关行为
def getUserIdData():
    idList=loginData['id'].unique()
    idData=pd.DataFrame({'id':idList},index=idList)
    return idData


#对登录数据的字段组进行聚合处理
def  aggLoginDataById(userData,loginData,columns,aggFunNames):
    if isinstance(columns,(str)):
        columns=[columns]
    if isinstance(aggFunNames,(str)):
        aggFunNames=[aggFunNames]
    id_cols=columns#.copy()
    id_cols.append('id')
    # print(id_cols)
    t=loginData[id_cols].groupby(loginData['id'])[columns].agg(aggFunNames)
    userData=pd.concat([userData, t], axis=1)
    return userData


#%%time
#登录的时间的处理
def loginDataTimeInit(loginData):
#     loginData['ts']=loginData['time']#.dt.second
#     print(loginData.head(2))
    t=loginData[['id','time']].sort_values(by='time').groupby('id')['time'].diff()
    loginData['login_diff_day']=t.dt.days
    loginData['login_diff_seconds']=t.dt.seconds
    return loginData


#对登录数据的处理
def loginDataInit(loginData):
    loginData=loginData.copy()
    userData=getUserIdData()
    userData=aggLoginDataById(userData,loginData,['device','log_from','city','result','type'],'nunique')
    userData=aggLoginDataById(userData,loginData,'timelong',['min','max','std','var','mean','skew'])
    loginData=loginDataTimeInit(loginData)
    userData=aggLoginDataById(userData,loginData,['login_diff_day','login_diff_seconds'],['min','max','std','var','mean','skew','mad'])
#     print(userData.head(2))
    loginData=pd.merge(loginData,userData,on='id',how='inner')
    del loginData['device'],loginData['log_from'],loginData['ip'],loginData['city']
    del loginData['result'],loginData['timestamp'],loginData['type']
    return loginData


#对交易数据的处理
def tradeDataInit(tradeData):
    tradeData=tradeData.copy()
    t=tradeData[['id','time']].sort_values(by='time').groupby('id')['time'].diff()
    tradeData['trade_diff_time']=t.dt.seconds
    return tradeData


#对登录和交易的融合的数据的处理，返回x,y
def allDataInit(loginData,tradeData):
    allData=pd.merge(tradeData,loginData,on='log_id',how='inner')
    del allData['log_id'],allData['id_x'],allData['id_y']
    del allData['time_x'],allData['time_y']
    allData.info()
#     del allData['is_sec'],allData['type'],allData['log_from'],allData['result'],allData['city']
#     allData.info()
    x=allData.iloc[:,2:].values
    y=allData['is_risk'].values
    return x,y


from sklearn.metrics import fbeta_score
#评估函数
def rocJdScore(*args):
    from sklearn import metrics
    return metrics.make_scorer(fbeta_score,beta=0.1, greater_is_better=True)(*args)


#生成训练用的pipline
def getPipe():
    # 下面，我要用逻辑回归拟合模型，并用标准化和PCA（30维->2维）对数据预处理，用Pipeline类把这些过程链接在一起
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.pipeline import Pipeline
    import xgboost as xgb
    from xgboost.sklearn import XGBClassifier
    #xgb的配置
    xgbFier = XGBClassifier(
             learning_rate =0.3,
             n_estimators=1000,
             max_depth=5,
             min_child_weight=1,
             gamma=0,
             subsample=0.8,
             colsample_bytree=0.8,
             objective= 'binary:logistic',
             nthread=2,
             scale_pos_weight=1,
             seed=27,
             silent=0
    )
    # 用StandardScaler和PCA作为转换器，LogisticRegression作为评估器
    estimators = [
#         ('scl', StandardScaler()), 
#                   ('pca', PCA(n_components=2)), 
#                    ('rf', RandomForestClassifier(random_state=1,
#                                                  max_depth= 50,
#                                                  min_samples_leaf= 3,
#                                                  min_samples_split= 10,
#                                                  n_estimators= 20,
#                                                 )),
#                   ('dtc',DecisionTreeClassifier(criterion='entropy')),
                                    ('xgb',xgbFier),
#                   ('lr', LogisticRegression())
                 ]
    # estimators = [ ('clf', RandomForestClassifier(random_state=1))]
    # Pipeline类接收一个包含元组的列表作为参数，每个元组的第一个值为任意的字符串标识符，
    #比如：我们可以通过pipe_lr.named_steps['pca']来访问PCA组件;第二个值为scikit-learn的转换器或评估器
    pipe_lr = Pipeline(estimators)
    return pipe_lr

#%%time
#k-fold交叉验证

from sklearn.cross_validation import cross_val_score
pipe_lr=getPipe()
trainLoginData=loginDataInit(loginData)
trainTradeData=tradeDataInit(tradeData)
X_train,y_train=allDataInit(trainLoginData,trainTradeData)
#记录程序运行时间
import time 
start_time = time.time()
scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=3, n_jobs=1,scoring=rocJdScore)
print(scores)
# #整体预测
# X_train,y_train=getTrainData(isUndersample=False)
# pipe_lr
#输出运行时长
cost_time = time.time()-start_time
print("交叉验证 success!",'\n',"cost time:",cost_time,"(s)")

