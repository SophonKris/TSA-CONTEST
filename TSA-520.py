
# coding: utf-8

# 思路：
# 
# 拼接所有能合并的特征，链接statead，利用贝叶斯处理user缺省值
# 
# 其他的能拆就拆，然后全部one-hot
# 
# 对于AppID，先把一个用户的AppID连接在一起，然后使用tf-idf处理，得到App特征
# 
# 上下两个合起来，裸跑LogisticRegression，生成submission，跑全量数据（17-28）在线上30%评分在0.10305左右。
# 
# 没有代码重构

# In[1]:

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import numpy as np
import pickle
import math


# In[2]:

#评分函数
import scipy as sp
def logloss(act, pred):
  epsilon = 1e-15
  pred = sp.maximum(epsilon, pred)
  pred = sp.minimum(1-epsilon, pred)
  ll = -sp.mean(act*sp.log(pred) + sp.subtract(1,act)*sp.log(1-pred))
  return ll


# In[4]:

train = pd.read_csv('./pre/train.csv')
test = pd.read_csv('./pre/test.csv')

train.head()


# In[25]:

# statead = pd.read_csv('./statead.csv')
# statead.head()


# In[27]:

test.head()


# In[5]:

#时间离散化
train['clickTime_day'] = train['clickTime'].map(lambda x:int(x/10000))
train['clickTime_hour'] = train['clickTime'].map(lambda x:int(x/100%100))
train['clickTime_minute'] = train['clickTime'].map(lambda x:int(x%100))


# In[6]:

train.groupby(['clickTime_day'])['label'].value_counts()


# In[108]:

# 将第28天作为验证集  （集第一次更新）
# proof = train[train.clickTime_day==28]
train = train[(train.clickTime_day>=17 ) & (train.clickTime_day<= 28)]


# In[7]:

print test.shape,train.shape


# In[8]:

#时间离散化
test['clickTime_day'] = test['clickTime'].map(lambda x:int(x/10000))
test['clickTime_hour'] = test['clickTime'].map(lambda x:int(x/100%100))
test['clickTime_minute'] = test['clickTime'].map(lambda x:int(x%100))


# In[9]:

test.groupby(['clickTime_hour'])['label'].value_counts()


# In[3]:

#position直接加上去，LogisticRegression Logistic回归
#的训练得分 0.120106201117，可见position特征用处不大
position = pd.read_csv('./pre/position.csv')
# train = pd.merge(train,statead,position,on='positionID',how='left')
# test = pd.merge(test,statead,position,on='positionID',how='left')
position.head()


# In[26]:

# train = pd.merge(train,statead,on='creativeID',how='left')
# test = pd.merge(test,statead,on='creativeID',how='left')


# In[23]:

# statead.head()


# In[73]:

#numberical feature数字特征，feature_name总特征，categorical_feature 分类特征
#我们去掉label，convertiontime跑一次
feature_name = [a for a in train.columns if a not in ['label','conversionTime']]
categorical_feature = ['creativeID','userID','positionID','connectionType','telecomsOperator']


# In[74]:

#去掉除label，convertiontime的第二次数据集
train_label = train['label']
train = train[feature_name]
test_label = test['label']
test = test[feature_name]


# In[75]:

#添加appID特征（tfidf）
user_installedapps = pd.read_csv('./pre/user_installedapps.csv')
user_installedapps_count = user_installedapps.groupby('userID').agg(len).reset_index()#计数特征


# In[76]:

user_installedapps.head()


# In[77]:

user_installedapps_count.columns = ['userID','user_appID_count']
#2798058	app360 app361 app362 app375 app480 app481 app4  相当于app+value
user_installedapps = user_installedapps.groupby('userID').agg(lambda x:' '.join(['app'+str(s) for s in x.values])).reset_index()



# In[78]:

user_id_all = pd.concat([train.userID,test.userID],axis=0)
user_id_all = pd.DataFrame(user_id_all,columns=['userID'])
user_id_all.head()



# In[79]:

#不同用户的先提取出来
user_installedapps = pd.merge(user_id_all.drop_duplicates(),user_installedapps,on='userID',how='left')
user_installedapps = user_installedapps.fillna('Missing')
#至此，user_installedapps处理完毕


# In[80]:


tfv = TfidfVectorizer()
tfv.fit(user_installedapps.appID)



# In[81]:

#按照顺序转化为tfidf特征
user_installedapps = pd.merge(user_id_all,user_installedapps,on='userID',how='left')
user_installedapps = user_installedapps.fillna('Missing')
user_installedapps_tfv = tfv.transform(user_installedapps.appID)


# In[22]:

#保险起见，爱你，就储存吧
user_installedapps.to_csv('./pre/user-app.csv',index=None)


# In[11]:

def featureManipulation(dtfm, colList, func):
    '''依次处理某一dataframe内__所有__col的__所有__零值'''
    for col in colList:
        pr_col = func(dtfm, col)
        for row in pr_col.iterrows():
            zeroSample = dtfm[col][(dtfm[col] == 0)]
            replace = row[0]
            num = row[1][col].astype(int)
            if num > len(zeroSample):
                print(replace)
                num = len(zeroSample)
            if num <= 0:
                continue
            smpl = zeroSample.sample(num)
            smpl = smpl.replace(0, replace)
            dtfm[col].update(smpl)
    print(dtfm)


# In[12]:

# 这里是对user的例子
user = pd.read_csv('./pre/user.csv')
user.head()
def sln(dtfm, col):
    dtfm_col = dtfm[dtfm[col] > 0]
    pr_col = dtfm_col[col].value_counts()/len(dtfm_col[col])
    pr_col *= len(dtfm[col][(dtfm[col] == 0)])
    pr_col = pr_col.apply(np.round)
    pr_col = pr_col.to_frame()
    return pr_col
featureManipulation(user, ['age','gender','education','hometown','residence'], sln)


# In[13]:

user.isnull().values.any()


# In[ ]:

user['hometown_city'] = user['hometown']%100
user['hometown_province'] = (user['hometown']/100).astype('int')
user['residence_city'] = user['residence']%100
user['residence_province'] = (user['residence']/100).astype('int')


# In[4]:

ad = pd.read_csv('./pre/ad.csv')
ad.head()


# In[84]:

#合并特征
train = pd.merge(train,user_installedapps_count,on='userID',how='left')
train = pd.merge(train,user,on='userID',how='left')
train = pd.merge(train,ad,on='creativeID',how='left')



# In[85]:

#验证集合并特征
test = pd.merge(test,user_installedapps_count,on='userID',how='left')
test = pd.merge(test,user,on='userID',how='left')
test = pd.merge(test,ad,on='creativeID',how='left') 


# In[86]:

train.head()


# In[87]:

#保险起见，爱你，就储存吧
train.to_csv('./pre/train28.csv',index=None)


# In[88]:

test.head()


# In[89]:

#保险起见，爱你，就储存吧
test.to_csv('./pre/test26.csv',index=None)


# In[90]:

train.dtypes


# In[91]:

train = train.fillna(0)
test = test.fillna(0)
train.dtypes


# In[92]:

train_user_appID_count =  train[['user_appID_count']]
test_user_appID_count =  test[['user_appID_count']]
del train['user_appID_count'],test['user_appID_count']


# In[93]:

oneEnc = OneHotEncoder()
data_one = pd.concat([train,test])
data_one = oneEnc.fit_transform(data_one)
train_one = data_one[:train.shape[0]]
test_one = data_one[train.shape[0]:]


# In[94]:

print train_one.shape
print user_installedapps_tfv[:train.shape[0]].shape
print train_user_appID_count.shape
print train.shape


# In[95]:

train_user_appID_count.values


# In[100]:

train = hstack([train_one,user_installedapps_tfv[:train.shape[0]]])
test = hstack([test_one,user_installedapps_tfv[train.shape[0]:]])


# In[ ]:

# #输出训练集和测试集
# with open('train.pkl','w') as f:
#     pickle.dump(train,f)
# with open('test.pkl','w') as f:
#     pickle.dump(test,f)
# #读取训练集和测试集
# with open('train.pkl','r') as f:
#     train = pickle.load(f)
# with open('test.pkl','r') as f:
#     test = pickle.load(f)


# In[97]:

from sklearn.linear_model import LogisticRegression
print 'LogisticRegression Logistic回归'
lr = LogisticRegression(n_jobs=-1,random_state=2017)
lr.fit(train,train_label)
pred = lr.predict_proba(train)[:,1]
print '训练得分',logloss(train_label,pred)
# pred = lr.predict_proba(test)[:,1]
# print '验证得分',logloss(test_label,pred)


# In[101]:

# 取消了test线下测试
results = lr.predict_proba(test)[:,1]
# print '验证得分',logloss(test_label,pred)


# In[103]:

#输出结果
# a = pd.DataFrame({'instanceID':pd.read_csv('./pre/test.csv')['instanceID'],'prob':pred})
#输出
test1 = pd.read_csv('./pre/test.csv')
test1['prob'] = results
test1= test1[['instanceID','prob']]
test1.to_csv('./pre/submission.csv',index=None)
submission =  pd.read_csv('./pre/submission.csv') 


# In[ ]:

#import os
#os.system('shutdown -s')


# In[104]:

submission

