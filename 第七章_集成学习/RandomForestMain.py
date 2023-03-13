from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import tree
import time
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve,auc
from scipy import interp
import matplotlib.ticker as ticker
import csv
from sklearn.ensemble import RandomForestClassifier
#将数据转化为列表
def load_data_feature(file):
    data = []
    with open(file,'r',encoding='utf-8') as f:
        for lines in f.readlines():
            lines = list(lines.split())
            a = []
            for line in lines:              
                a.append(float(line))
            data.append(a)
    data = np.array(data)
    return data

def load_data_label(file):
    data = []
    with open(file,'r',encoding='utf-8') as f:
        for lines in f.readlines():
            data.append(int(lines))
    data = np.array(data)
    return data

def handle_data(data):
    for i in range(len(data)):
            if data[i] == 0 :
                data[i] =-1

#调库实现决策树
def build_tree(xtrain,ytrain,xtest,ytest):
    clf=tree.DecisionTreeClassifier(criterion = 'gini', max_depth=1) #以信息熵为参数
    clf.fit(xtrain,ytrain) #利用数据集进行训练
    score=clf.score(xtest,ytest) #进行测试 返回预测的准确度 
    print(score)

#构建基分类器
#data为数据集,利用KF.split划分训练集和测试集
def Forest(xtrain,ytrain,xtest,ytest):
    #定义n折交叉验证
    KF = KFold(n_splits = 5)

    auc_mean = []
    score_mean = []

    start = time.perf_counter()
    num_learners = 5

    for i in range(1,num_learners+1):
        aucs = []
        scores = []
        starti = time.perf_counter()
        for train_index,test_index in KF.split(xtrain,ytrain):
            #建立模型，并对训练集进行测试，求出预测得分
            #划分训练集和测试集
            X_train,X_test = xtrain[train_index],xtrain[test_index]
            Y_train,Y_test = ytrain[train_index],ytrain[test_index]
        
            model = RandomForestClassifier(n_estimators = i , criterion = 'gini')
            model.fit(X_train,Y_train)

            #利用model.predict获取测试集的预测值
            y_pred = model.predict_proba(X_test)

            #计算fpr(假阳性率),tpr(真阳性率),thresholds(阈值)[绘制ROC曲线要用到这几个值]
            fpr,tpr,threshold=roc_curve(Y_test ,y_pred[:,1] ,pos_label = 1)
            #计算auc
            roc_auc=auc(fpr,tpr)
            aucs.append(roc_auc)

            #计算模型分数
            score = model.score(xtest,ytest)
            scores.append(score)
            
        auc_mean.append(sum(aucs)/5.0)
        score_mean.append(sum(scores)/5.0)
        print("第%d个基学习器运行时间: %f s，AUC分数为：%f，准确度为：%f" % (i,time.perf_counter() - starti,auc_mean[i-1],score_mean[i-1]))
    
    print("RandomForest运行时间: %f s" % (time.perf_counter() - start))

    with open (file = 'RandomForestMain.csv',mode = 'w+',encoding='utf-8') as file:
            for i in range(num_learners):
                file.write(str(i+1)+','+str(auc_mean[i])+',')
                file.write(str(score_mean[i]))
                file.write('\n')

    print('写入完成！')

    



def plot_auc():
    #显示中文标签 
    plt.rcParams['font.sans-serif']=['SimHei'] 
    plt.rcParams['axes.unicode_minus']=False
    
    x = []
    auc_all= []
    with open (file = 'BoostMain.csv',mode = 'r+',encoding='utf-8') as file:
        lines = csv.reader(file)
        for line in lines:
            x.append(int(line[0]))
            auc_all.append(float(line[1]))
    
    x=np.array(x)
    auc_all=np.array(auc_all)
    np.round(auc_all,2)
    plt.title('AdaBoost')

    # 设置X, Y轴标题
    plt.xlabel('基学习器数量')
    plt.ylabel('AUC')
    plt.xlim(np.min(x)-0.05,np.max(x)+0.05)#确定横轴坐标范围
    plt.ylim(np.min(auc_all)-0.05,np.max(auc_all)+0.05)
    plt.plot(x,auc_all,'o') 
    plt.plot(x,auc_all,color='orange') 
    plt.show()
    plt.savefig('AdaBoost.png')

def main():
    #导入数据
    xtrain_file = './adult_dataset/adult_train_feature.txt'
    ytrain_file = './adult_dataset/adult_train_label.txt'
    xtest_file = './adult_dataset/adult_test_feature.txt'
    ytest_file = './adult_dataset/adult_test_label.txt'

    #划分训练集和验证集
    xtrain = []
    ytrain = []
    xtest = []
    ytest = []
    xtrain = load_data_feature(xtrain_file)
    ytrain = load_data_label(ytrain_file)
    xtest = load_data_feature(xtest_file)
    ytest = load_data_label(ytest_file)
    handle_data(ytrain)
    handle_data(ytest)
    
    Forest(xtrain,ytrain,xtest,ytest)
    plot_auc()
main()