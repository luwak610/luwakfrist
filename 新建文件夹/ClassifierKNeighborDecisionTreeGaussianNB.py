import pandas as pd
import numpy as np  
 
from sklearn.preprocessing import Imputer#预处理模块
from sklearn.model_selection import train_test_split #cross_validation是过时的交叉
from sklearn.metrics import classification_report#预测结果评估模块
   
from sklearn.neighbors import KNeighborsClassifier#K近邻分类器
from sklearn.tree import DecisionTreeClassifier#决策树分类器
from sklearn.naive_bayes import GaussianNB#高斯朴素贝叶斯函数
 
def load_datasets(feature_paths, label_paths):
    feature = np.ndarray(shape=(0,41))#特征文件矩阵列表,其列数为41,与特征维度一致
    label = np.ndarray(shape=(0,1))#标签文件矩阵列表
    for file in feature_paths:
        #使用逗号分隔符读取特征数据.将?替换标记为缺失值.文件中不包含表头,即原文本中就是用?表示缺失值的
        df = pd.read_table(file, delimiter=',', na_values='?', header=None)
        #使用平局值补全缺失值,然后将数据进行补全
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(df)#生成预处理前
        df = imp.transform(df)#得到预处理结果
        #将新读入的数据合并到特征集合中
        feature = np.concatenate((feature, df))
     
    for file in label_paths:
        #读取标签数据,文件不包括表头,由于标签文件没有缺失值所以不用预处理
        df = pd.read_table(file, header=None)
        #将新都如的数据合并到标签集合中
        label = np.concatenate((label, df))     
       #将标签规整为1维向量 
    label = np.ravel(label)
    return feature, label
 
if __name__ == '__main__':
    ''' 数据路径 '''
    featurePaths = ['feeddata/A/A.feature','feeddata/B/B.feature','feeddata/C/C.feature','feeddata/D/D.feature','feeddata/E/E.feature']
    labelPaths = ['feeddata/A/A.label','feeddata/B/B.label','feeddata/C/C.label','feeddata/D/D.label','feeddata/E/E.label']
    ''' 读入数据  '''
    #将前4个数据作为训练集读入
    x_train,y_train = load_datasets(featurePaths[:4],labelPaths[:4])
    #将最后1个数据作为测试集读入
    x_test,y_test = load_datasets(featurePaths[4:],labelPaths[4:])
    #使用全数据作为训练集,借助train_test_split函数将训练数据打乱,test_size验证集占训练集0%，
    x_train, x_, y_train, y_ = train_test_split(x_train, y_train, test_size = 0.0)
     
    print('Start training knn')
    #创建K近邻分类器,并在测试集上进行预测
    knn = KNeighborsClassifier().fit(x_train, y_train)
    print('Training done')
    answer_knn = knn.predict(x_test)
    print('Prediction done')
    #创建决策树分类器,并在测试集上进行预测 
    print('Start training DT')
    dt = DecisionTreeClassifier().fit(x_train, y_train)
    print('Training done')
    answer_dt = dt.predict(x_test)
    print('Prediction done')
    #创建贝叶斯分类器,并在测试集上进行预测 
    print('Start training Bayes')
    gnb = GaussianNB().fit(x_train, y_train)
    print('Training done')
    answer_gnb = gnb.predict(x_test)
    print('Prediction done')
     #计算准确率和召回率,f1 f1-score,和支持度support四个维度进行衡量
    #f1看作是模型准确率和召回率的一种加权平均
    print('\n\nThe classification report for knn:')
    print(classification_report(y_test, answer_knn))
    print('\n\nThe classification report for DT:')
    print(classification_report(y_test, answer_dt))
    print('\n\nThe classification report for Bayes:')
    print(classification_report(y_test, answer_gnb))