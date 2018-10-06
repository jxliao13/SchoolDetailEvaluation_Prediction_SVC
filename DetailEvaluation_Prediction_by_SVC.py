
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame


# In[2]:


#pd.set_option('display.width', 1000, 'display.max_rows', 1000)
pd.options.display.max_columns = None
#import numpy as np
np.set_printoptions(threshold=np.inf)


# GOAL: Predict Classify of [Is_R] (預測詳評結果是否需補強)
# 
# 1.讀取csv檔案  2.檢查是否存在Null值 
# 
# 共28個 features: 'D_Dbuild','D_StructureSystem','D_structure','D_floor','D_floorTA','D_floorTAGround','D_1floorCorridorColA','D_1floorClassColA','D_1floorInsideColA',,'D_basintype','D_I','D_Demand','D_Tx','D_Ty','D_sds','D_CLlarge','D_CLsmall','D_MaxCl','D_NeutralDepth','T_edead','T_elive','T_height','T_floorA','AVG_confc','AVG_MBfy','AVG_stify'
# 
# 分類依據: 'Is_R'

# In[3]:


#[1]讀取自SQL轉存之csv檔
#資料庫中已做第一階段前處理
df = pd.read_csv("DE_result_ver3.csv", low_memory=False)
features = ['D_Dbuild','D_StructureSystem','D_structure','D_floor','D_floorTA','D_floorTAGround','D_1floorCorridorColA','D_1floorClassColA','D_1floorInsideColA','D_basintype','D_I','D_Demand','D_Tx','D_Ty','D_XRCwallA','D_YRCwallA','D_sds','D_CLlarge','D_CLsmall','D_MaxCl','D_NeutralDepth','T_edead','T_elive','T_height','AVG_confc','AVG_MBfy','AVG_stify','Is_R']
df = df[features]

#[2]檢查NULL值
#統計null值數量
print("總共資料數:",df.shape)
print("有無NULL:",df.isnull().values.any())
print(df.isnull().sum())

#-------將null用平均值取代-----------
#df['D_Demand'] = df['D_Demand'].replace(np.nan,df['D_Demand'].mean())



# 3.處理類別特徵，轉成數值 by One-hot encoding

# In[4]:


#[3-1]處理分類-D_StructureSystem
df2 = pd.get_dummies(df['D_StructureSystem'], prefix='StrucSys')
df = df.drop('D_StructureSystem',1)
df = pd.concat([df2,df],axis=1)
df



# In[5]:


#[3-2]處理分類-D_Structure
df3 = pd.get_dummies(df['D_structure'], prefix='Struc')
df = df.drop('D_structure',1)
df = pd.concat([df3,df],axis=1)
df


# In[6]:


#[3-3]處理分類-D_basintype 地盤種類
df4 = pd.get_dummies(df['D_basintype'], prefix='Base')
#df4.head()
#print(df4)
#df = df.join(df4).head()
df = df.drop('D_basintype',1)
df = pd.concat([df4,df],axis=1)
df


# 4.Describe資料分布

# In[7]:


#[4]觀察資料分布
print(df.describe())


# 5.清除outlier之前-資料散佈圖

# In[8]:


def scatter(x,y,data1_name,data2_name):
    #plt.set_title('Data Analysis')  
    plt.xlabel(data1_name, fontsize=16)
    plt.ylabel(data2_name, fontsize=16)
    plt.scatter(x, y, marker='o', alpha= 0.3 , color = 'blue')
    plt.show()

#觀察資料處理前-屬性 vs label視覺化
for i in range(0,len(df.columns)):
    data1 = df.iloc[:,i]
    data2 = df.iloc[:,len(df.columns)-1]
    data1_name = df.columns[i]
    data2_name = df.columns[len(df.columns)-1]
    scatter(data1, data2,data1_name,data2_name)


# 5-2.依照上一步之結果評估並清除異常值

# In[9]:


#清除異常值
#要清除的欄位
clean_list = ['D_floorTA','D_floorTAGround','D_1floorCorridorColA','D_1floorClassColA','D_1floorInsideColA','D_Demand','D_Tx','D_Ty','D_XRCwallA','D_YRCwallA','D_sds','D_CLlarge','D_CLsmall','D_MaxCl','D_NeutralDepth','T_edead','T_elive','T_height','AVG_confc','AVG_MBfy','AVG_stify']
clean_num = 0
for i in range(0,len(clean_list)):
    #print(df.columns[i])
    col_name = clean_list[i]
    print(col_name)
    print("第",i+1," 次整理前資料數",df.shape)
    before =df.shape[0]
    df = df[np.abs(df[col_name]-df[col_name].mean())<=(3*df[col_name].std())]
    print("第",i+1," 次整理後資料數",df.shape)
    after =df.shape[0]
    print("清除",before-after,"筆outlier")
    clean_num = clean_num + before-after
print("-----------------")
print("總共清除了:"+str(clean_num)+"筆")


# 6.觀察處理後的數據分布統計

# In[10]:


print(df.shape)
df.describe()


# In[11]:


#觀察資料處理後屬性與label視覺化
# for i in range(0,len(df.columns)):
#     data1 = df.iloc[:,i]
#     data2 = df.iloc[:,len(df.columns)-1]
#     data1_name = df.columns[i]
#     data2_name = df.columns[len(df.columns)-1]
#     scatter(data1, data2,data1_name,data2_name)


# 6.清除Outlier PART2

# In[12]:


#------------第二輪----------
#清除異常值
#要清除的欄位
clean_list = ['D_floorTA','D_floorTAGround','D_1floorCorridorColA','D_1floorClassColA','D_1floorInsideColA','D_Tx','D_Ty','D_XRCwallA','D_YRCwallA','D_CLlarge','D_CLsmall','D_MaxCl','T_elive','AVG_confc','AVG_stify']
clean_num = 0
for i in range(0,len(clean_list)):
    #print(df.columns[i])
    col_name = clean_list[i]
    print(col_name)
    print("第",i+1," 次整理前資料數",df.shape)
    before =df.shape[0]
    df = df[np.abs(df[col_name]-df[col_name].mean())<=(3*df[col_name].std())]
    print("第",i+1," 次整理後資料數",df.shape)
    after =df.shape[0]
    print("清除",before-after,"筆outlier")
    clean_num = clean_num + before-after
print("-----------------")
print("總共清除了:"+str(clean_num)+"筆")


# In[13]:


print(df.shape)
df.describe()


# In[14]:


#觀察資料處理後屬性與label視覺化
for i in range(0,len(df.columns)):
    data1 = df.iloc[:,i]
    data2 = df.iloc[:,len(df.columns)-1]
    data1_name = df.columns[i]
    data2_name = df.columns[len(df.columns)-1]
    scatter(data1, data2,data1_name,data2_name)


# In[15]:


#資料欄位
#df.columns
columns = ['Base_1', 'Base_2', 'Base_3', 'Base_4', 'Struc_1', 'Struc_2', 'Struc_3',
       'Struc_4', 'Struc_5', 'Struc_6', 'StrucSys_1', 'StrucSys_2',
       'StrucSys_3', 'StrucSys_4', 'D_Dbuild', 'D_floor', 'D_floorTA',
       'D_floorTAGround', 'D_1floorCorridorColA', 'D_1floorClassColA',
       'D_1floorInsideColA', 'D_I', 'D_Demand', 'D_Tx', 'D_Ty', 'D_XRCwallA',
       'D_YRCwallA', 'D_sds', 'D_CLlarge', 'D_CLsmall', 'D_MaxCl',
       'D_NeutralDepth', 'T_edead', 'T_elive', 'T_height', 'AVG_confc',
       'AVG_MBfy', 'AVG_stify']
len(columns)


# 7.將Data分為訓練用及測試用

# In[16]:


#保留30%當作測試用資料
X_train, X_test, y_train, y_test = train_test_split(
    df[['Base_1', 'Base_2', 'Base_3', 'Base_4', 'Struc_1', 'Struc_2', 'Struc_3',
       'Struc_4', 'Struc_5', 'Struc_6', 'StrucSys_1', 'StrucSys_2',
       'StrucSys_3', 'StrucSys_4', 'D_Dbuild', 'D_floor', 'D_floorTA',
       'D_floorTAGround', 'D_1floorCorridorColA', 'D_1floorClassColA',
       'D_1floorInsideColA', 'D_I', 'D_Demand', 'D_Tx', 'D_Ty', 'D_XRCwallA',
       'D_YRCwallA', 'D_sds', 'D_CLlarge', 'D_CLsmall', 'D_MaxCl',
       'D_NeutralDepth', 'T_edead', 'T_elive', 'T_height', 'AVG_confc',
       'AVG_MBfy', 'AVG_stify']], df[['Is_R']], test_size=0.3, random_state=0)


# 8.將Data標準化

# In[17]:


#數據標準化
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


# 9.SVC

# In[18]:


# "Support vector classifier"
from sklearn.svm import SVC 
svm = SVC(kernel='rbf', probability=True)
svm.fit(X_train_std,y_train['Is_R'].values)


# In[19]:


#svm.predict(X_test_std)
y_predict = svm.predict(X_test_std)


# In[20]:


#測試資料集之結果
y_test['Is_R'].values


# In[21]:


#預測資料集之結果
print(y_predict)


# 10.統計預測錯誤數

# In[22]:


#預測錯誤之數量統計
error = 0
for i, v in enumerate(svm.predict(X_test_std)):
    if v!= y_test['Is_R'].values[i]:
        error+=1
print(error)


# In[23]:


#svm.predict_proba(X_test_std)


# In[24]:


#查看正確率
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_predict)
print("Accuracy Rate:",accuracy)


# In[25]:


#查看auc值進行效能評估
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict)
auc = metrics.auc(fpr, tpr)
print("Auc:",auc)


# 11.尋找最佳參數組合

# In[26]:


#測試參數組合
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
#GridSearchCV是建立一個dictionary來組合要測試的參數
param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}

#GridSearchCV算是一個meta-estimator，參數中帶有estimator，像是SVC，重點是會創造一個新的estimator，但又表現的一模一樣。也就是estimator=SVC時，就是作為分類器
#Verbose可設定為任一整數，它只是代表數字越高，文字解釋越多
grid = GridSearchCV(SVC(),param_grid,verbose=3)

#利用剛剛設定的參數來找到最適合的模型
grid.fit(X_train_std,y_train['Is_R'].values)

#顯示最佳參數組合
grid.best_params_

#顯示最佳estimator參數
grid.best_estimator_

#利用剛剛的最佳參數再重新預測測試組
grid_predictions = grid.predict(X_test_std)

#評估新參數的預測結果好壞

print(confusion_matrix(y_test,grid_predictions))
print('\n')
print(classification_report(y_test,grid_predictions))


# In[27]:


#顯示最佳參數調整C及gamma值
print(grid.best_estimator_)


# In[28]:


print(grid_predictions)


# In[29]:


#查看正確率
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, grid_predictions)
print("最佳化後, Accuracy Rate:",accuracy)

#查看auc值進行效能評估
fpr, tpr, thresholds = metrics.roc_curve(y_test, grid_predictions)
auc = metrics.auc(fpr, tpr)
print("最佳化後, Auc:",auc)


# In[30]:


#顯示混淆矩陣

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test,grid_predictions)
sns.heatmap(mat.T,square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()


# 參考連結:
# 
# 1.[資料分析&機器學習] 第3.4講：支援向量機(Support Vector Machine)介紹https://medium.com/@yehjames/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-4%E8%AC%9B-%E6%94%AF%E6%8F%B4%E5%90%91%E9%87%8F%E6%A9%9F-support-vector-machine-%E4%BB%8B%E7%B4%B9-9c6c6925856b
# 
# 2.Python學習筆記#16：機器學習之SVM實作篇 
# http://psop-blog.logdown.com/posts/3150995-python-machine-learning-svm
# 
# 3.機器學習（6）隨機森林與支持向量機 https://ithelp.ithome.com.tw/articles/10187569
