#coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#log_id,id转换为字符串
loginData=pd.read_csv("input/t_login.csv",dtype={'log_id':str,'id':str})
loginTestData=pd.read_csv("input/t_login_test.csv",dtype={'log_id':str,'id':str})
tradeData=pd.read_csv("input/trade_2_login.csv",dtype={'id':str,'login_id':str})
tradeTestData=pd.read_csv("input/t_trade_test_2_login.csv",dtype={'id':str,'login_id':str})
loginTestData=loginTestData.append(loginData)

#得到DataFrame中的无重复的id
def getUserIDFromDataFrame(dataFrame):
    return pd.DataFrame({'id':dataFrame['id'].unique()})
	
#得到登录中所用时间为整秒的计算
def getTimelongScale(loginData,test):
    #目的是得出 是否为整数秒的True,False两列，每个id有多少个True多少个False，按id排序
    #计算是否为整秒
    loginData['timeScale']=(loginData['timelong']%1000==0)
    #分别为整数 ，不为整数的次数
    tmp = pd.concat([loginData['id'],pd.get_dummies(loginData['timeScale'],prefix = 'timeScale')],1).groupby('id').sum().reset_index()
    #gd=loginData['timeScale'].groupby([loginData['id'],loginData['timeScale']])
    #tmp=gd.count().unstack().reset_index()
    #tmp.to_csv('output/tmp.csv')
    #tmp=tmp.fillna(0)
    test=pd.merge(tmp,test,on='id',how='inner')
    del loginData['timeScale'],tmp
    #tmp删掉，减小内存
    return test


#得到某列的分组不同的个数
def getCountsByColumnName(loginData,test,columnName):
    num=loginData[['id',columnName]].groupby(loginData['id'])[columnName].nunique()
    t=pd.DataFrame(num)
    t.insert(0, 'id', num.index)
    t.columns=['id',columnName+'_Counts']
    test=pd.merge(t,test,on='id',how='inner')
    return test

#得到某列的分组平均值
def getMeanByColumnName(loginData,test,columnName):
    t=loginData[columnName].groupby(loginData['id'])
    t=t.mean().reset_index()
    t.columns=['id',columnName+'_Mean']
    test=pd.merge(t,test,on='id',how='inner')
    return test

#得到某列的分组最小值
def getMinByColumnName(loginData,test,columnName):
    t=loginData[columnName].groupby(loginData['id'])
    t=t.min().reset_index()
    t.columns=['id',columnName+'_min']
    test=pd.merge(t,test,on='id',how='inner')
    return test

#得到某列的分组最大值
def getMaxByColumnName(loginData,test,columnName):
    t=loginData[columnName].groupby(loginData['id'])
    t=t.max().reset_index()
    t.columns=['id',columnName+'_max']
    test=pd.merge(t,test,on='id',how='inner')
    return test


#得到某列的分组方差
def getVarByColumnName(loginData,test,columnName):
    t=loginData[columnName].groupby(loginData['id'])
    t=t.var().reset_index()
    t.columns=['id',columnName+'_var']
    test=pd.merge(t,test,on='id',how='inner')
    return test

#得到某列的分组标准差
def getStdByColumnName(loginData,test,columnName):
    t=loginData[columnName].groupby(loginData['id'])
    t=t.std().reset_index()
    t.columns=['id',columnName+'_std']
    test=pd.merge(t,test,on='id',how='inner')
    return test

#得到对登录result的处理
def getLoginResultCount(loginData,test):
    r31=loginData[loginData['result']!=1]
    t=r31['result'].groupby(r31['id'])
    t=t.count().reset_index()
    t.columns=['id','result_no1_Counts']
    test=pd.merge(t,test,on='id',how='inner')
    test=test.fillna(0)
    return test


#每次登陆的时间段的比例
def getLoginHoursSacle(tradeData,loginData):
    loginData['tx']=pd.to_datetime(loginData['time'])
    loginData['date']=loginData['tx'].dt.date
    loginData['hour']=loginData['tx'].dt.hour
    #每天在同一个小时内多次登陆只算一次
    un=loginData[['id','date','hour','log_id']].groupby(by=['id','date','hour'])['log_id'].nunique()
    un=un.reset_index()
    un.to_csv('output/un.csv')
    tmp=un[['id','hour']].groupby(by='id')['hour'].agg([np.mean,np.std,np.var]).reset_index()    
    tmp.columns=['id','l_time_mean','l_time_std','l_time_var']
    #tmp.to_csv('output/tmpTrain.csv')
    loginDataTime=pd.merge(loginData,tmp,on='id',how='inner')
    loginDataTime['sub_l_time_mean']=np.abs(loginDataTime['hour']-loginDataTime['l_time_mean']) 
    tradeData=pd.merge(tradeData,loginDataTime[['log_id','l_time_mean','l_time_std','l_time_var','sub_l_time_mean']],
                       left_on='login_id',right_on='log_id',how='inner')
    del tradeData['log_id']
    del loginDataTime
    del loginData['tx'],loginData['date'],loginData['hour'],un,tmp
    #因为线上分数有点低，暂时删除以下
    del tradeData['l_time_mean'],tradeData['l_time_std'],tradeData['l_time_var']
    return tradeData
# getLoginHoursSacle(tradeData,loginData)

from sklearn.preprocessing import OneHotEncoder
#处理交易数据
def tradeDataInit(tradeData,loginData):
    tradeData=tradeData.copy()
    tradeData=getLoginHoursSacle(tradeData,loginData)
    #构建一个小时的
    tradeData['hours']=tradeData['time'].str.extract('\\s(\\d\\d):')
    tradeData['hours']=tradeData['hours'].astype('int')
    tradeData=pd.merge(tradeData,loginData[['log_id','time']],left_on='login_id',right_on='log_id',how='inner')
    del tradeData['log_id']
#     tradeData['hours']=
#     print(OneHotEncoder(sparse = False).fit_transform( tradeData[['hours']]))
    #交易时间
    tradeData['tx']=pd.to_datetime(tradeData['time_x'])
    #登录时间
    tradeData['ty']=pd.to_datetime(tradeData['time_y'])
    #交易时间和登录时间之间的差值
    tradeData['st']=(tradeData['tx']-tradeData['ty']).dt.seconds
    #每次交易时间的差
    tradeData['trade_time_sub']=tradeData[['id','tx']].sort_values(by='tx').groupby('id').diff()['tx']
    tradeData['trade_time_sub_day']=tradeData['trade_time_sub'].dt.days
    tradeData=getMaxByColumnName(tradeData,tradeData,'trade_time_sub_day')
    tradeData=getMeanByColumnName(tradeData,tradeData,'trade_time_sub_day')
    tradeData=getMinByColumnName(tradeData,tradeData,'trade_time_sub_day')
    tradeData=getStdByColumnName(tradeData,tradeData,'trade_time_sub_day')
    tradeData=getVarByColumnName(tradeData,tradeData,'trade_time_sub_day')
    #使用的比例
    tradeData=getTradeLoginColumScale(tradeData,loginData,'device')
    tradeData=getTradeLoginColumScale(tradeData,loginData,'city')
    
    del tradeData['tx'],tradeData['ty'],tradeData['time_y'],tradeData['login_id'],tradeData['trade_time_sub']
    tradeData.rename(columns={'time_x':'time'},inplace=True)
    return tradeData

def getTradeLoginColumScale(tradeData,loginData,cname):
    #计算cname列上分别使用的个数
    c_count=loginData[['id',cname,'log_id']].groupby(['id',cname]).count()
    c_count=c_count.reset_index()
    c_count.columns=['id',cname,cname+'_count_tmp']
    #使用的总个数
    c_sum=c_count[['id',cname+'_count_tmp']].groupby('id').sum().reset_index()
    c_sum.columns=['id',cname+'_count_sum']
    c=pd.merge(c_count,c_sum,on='id',how='inner')
    #使用的比例
    c[cname+'_scale']=c[cname+'_count_tmp']/c[cname+'_count_sum']
    del c[cname+'_count_tmp'],c[cname+'_count_sum']
    c=pd.merge(loginData[['log_id','id',cname]],c,on=['id',cname],how='inner')
    tradeData=pd.merge(tradeData,c[['log_id',cname+'_scale']],left_on='login_id',right_on='log_id',how='inner')
    del c,tradeData['log_id']
    return tradeData

#把处理后的登录数据和交易数据内连接
def createAllData(test,tradeData):
    tmp=pd.merge(test,tradeData,on='id',how='inner')
#     tmp=tmp.fillna(0)
#     print(tmp.info())
    return tmp

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

#得到训练用的测试集元组（x，y）
def getTrainData(isUndersample=False):
    allData=transferData(loginData,tradeData)
    if(isUndersample):
        #进行undersample的处理
        number_records_fraud=len(allData[allData['is_risk']==1]) #有风险的个数
        fraud_indices=np.array(allData[allData['is_risk']==1].index) #有风险的index
        normal_indices=allData[allData['is_risk']==0].index
        random_normal_indices=np.random.choice(normal_indices,number_records_fraud,replace=False)
        random_normal_indices=np.array(random_normal_indices)
        under_sample_indices=np.concatenate([fraud_indices,random_normal_indices])
        allData=allData.iloc[under_sample_indices,:]
    
    x=allData.iloc[:,3:].values
    y=allData['is_risk'].values
    
    return x,y

#生成学习算法
def jdPipeFited(pipe_lr):    
    x,y=getTrainData(isUndersample=False)
    from sklearn.cross_validation import train_test_split
    # 拆分成训练集(80%)和测试集(20%)
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=1) 
    
    pipe_lr.fit(x, y)
    return pipe_lr

'''
你可以理解为短时间内变化城市、切换IP、更换设备、三更半夜多笔交易、不在常用设备登陆、不在常在城市登陆、短时间多次登陆的后续交易
'''
#生成训练数据
def transferData(loginData,tradeData):
    print 'first loginDataColumns:',list(loginData)
    lData=getUserIDFromDataFrame(loginData)    
    #登录耗费时间段是否为整数
    lData=getTimelongScale(loginData,lData)
    print '\n-------------------:\nafter timeScalse:',list(lData)
    lData=getMinByColumnName(loginData,lData,'timelong')
    lData=getMaxByColumnName(loginData,lData,'timelong')
    lData=getMeanByColumnName(loginData,l Data,'timelong')
    lData=getVarByColumnName(loginData,lData,'timelong')
    lData=getStdByColumnName(loginData,lData,'timelong')
    print 'after five:',list(lData)
	# ##device的个数
	#lData=getCountsByColumnName(loginData,lData,'device')
	# ##ip的个数
	#lData=getCountsByColumnName(loginData,lData,'ip')
	#lData=getCountsByColumnName(loginData,lData,'log_from')
    
    lData=getCountsByColumnName(loginData,lData,'type')
	# #城市的个数
    lData=getCountsByColumnName(loginData,lData,'58')
    print 'after counts: ',list(lData)
    #登录的次数
	#lData=getCountsByColumnName(loginData,lData,'log_id')
    #平均值
#     lData=getMeanByColumnName5(loginData,lData,'device')
#     lData=getMeanByColumnName(loginData,lData,'log_from')
#     lData=getMeanByColumnName(loginData,lData,'ip')
#     lData=getMeanByColumnName(loginData,lData,'city')
#     lData=getMeanByColumnName(loginData,lData,'result')
#     lData=getMeanByColumnName(loginData,lData,'type')
    #result的处理
    lData=getLoginResultCount(loginData,lData)
    print 'after result: ',list(lData)
    tData=tradeData.copy()
    tData=tradeDataInit(tData,loginData)
    #tData.to_csv('output/tradeDataInit.csv')
	#tData=getLastSubTime(loginData,tData)
    #lData.to_csv('output/lDataC.csv')
    #tData.to_csv('output/tDataC.csv')
    allData=createAllData(lData,tData)    
    #allData.to_csv('output/allDataC.csv')
    del allData['time']
    # get a list of columns
    cols = list(allData)
    if 'is_risk' in allData.columns:
        cols.insert(0, cols.pop(cols.index('is_risk')))
    cols.insert(0, cols.pop(cols.index('rowkey')))
    
    allData = allData.ix[:, cols]
    #allData.to_csv('output/allDataDel.csv')
    print(allData.head(2))
    return allData


#k-fold交叉验证
def crossval():
	from sklearn.cross_validation import cross_val_score
	pipe_lr=getPipe()
	X_train,y_train=getTrainData(isUndersample=False)
	#记录程序运行时间
	import time 
	start_time = time.time()
	scores = cross_val_score(estimator=pipe_lr, X=X_train, y=y_train, cv=3, n_jobs=2,scoring=rocJdScore)
	print(scores)
	# #整体预测
	# X_train,y_train=getTrainData(isUndersample=False)
	# pipe_lr
	#输出运行时长
	cost_time = time.time()-start_time
	print("交叉验证 success!",'\n',"cost time:",cost_time,"(s)")	
	
#网格搜索实验
def gridSer():
	from sklearn.grid_search import GridSearchCV
	parameters = {
	#     'rf__n_estimators': (5, 10, 20, 50),
	#     'rf__max_depth': (50, 150, 250),
	#     'rf__min_samples_split': [10, 2, 3],
	#     'rf__min_samples_leaf': (1, 2, 3),
		#xgb的参数
		'xgb__max_depth':(4,6),
		'xgb__learning_rate':(0.3,0.5)

	}
	pipe_lr=getPipe()
	X_train,y_train=getTrainData()


	#网格搜索
	grid_search = GridSearchCV(pipe_lr, parameters, n_jobs=-1, verbose=1, scoring=rocJdScore)
	grid_search.fit(X_train, y_train)

	#获取最优参数
	print('最佳效果：%0.3f' % grid_search.best_score_)
	print('最优参数：')
	best_parameters = grid_search.best_estimator_.get_params()
	for param_name in sorted(parameters.keys()):
		print('\t%s= %r' % (param_name, best_parameters[param_name]))

	# #预测以及分类器参数报告
	# predictions = grid_search.predict(X_test)
	# print(classification_report(y_test, predictions))	
	
#学习曲线
def learncur():
	from sklearn.learning_curve import learning_curve

	pipe_lr = getPipe()
	X_train,y_train=getTrainData()
	# train_sizes参数指定用于生成学习曲线的训练集数量，如果是分数指的是相对数量，整数指的是绝对数量
	train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr, X=X_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=3,
															n_jobs=2,scoring=rocJdScore)
	print('==='*10)
	train_mean = np.mean(train_scores, axis=1)
	train_std = np.std(train_scores, axis=1)
	test_mean = np.mean(test_scores, axis=1)
	test_std = np.std(test_scores, axis=1)

	plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
	plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
	plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
	plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
	plt.grid()
	plt.xlabel('Number of training samples')
	plt.ylabel('Accuracy')
	plt.legend(loc='lower right')
	plt.ylim([0.0, 1.0])
	plt.savefig("curve.png")
	plt.show()
	
def pre():
	#预测
	pipe=getPipe()
	pipe=jdPipeFited(pipe)
	preData=transferData(loginTestData,tradeTestData)
	x_pred=preData.iloc[:,2:].values
	y_pred=pipe.predict_proba(x_pred)
	print '--------------------------'
	print 'y_pred.shape', y_pred.shape
	print y_pred

	print(np.sum(y_pred))
	t=y_pred[:,1]
	print '------------------------------'
	print 't.shape',t.shape
	print t
	t[t<0.99]=0
	t[t>=0.99]=1
	t
	y_pred=t
def result():
	p=pd.DataFrame(y_pred)
	subData=pd.DataFrame(preData['rowkey'])
	subData['is_risk']=p
	#之前用很多inner join，很多数据没有，都默认处理为没有风险
	subData=pd.merge(tradeTestData[['rowkey']],subData,on='rowkey',how='left')
	subData=subData.fillna(0)
	subData['is_risk']=subData['is_risk'].astype('int')

	subData.to_csv('output/sub.csv',header=False,index=False)
def getData():
	x,y=getTrainData(isUndersample=False)
	#preData=transferData(loginTestData,tradeTestData)
	#x_pred=preData.iloc[:,2:].values
def preDat():
	preData=transferData(loginTestData,tradeTestData)
	preData.to_csv('output/xtest.csv',header = True, index = True)
	col = preData.columns
	df = pd.DataFrame(col)
	df.to_csv('output/xtestColumns.csv')
print 'start'
getData()
print 'over'
