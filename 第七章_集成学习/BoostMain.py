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
class TreeClassifier_weight:
    def __init__(self):
        self.best_error = 1
        self.best_feature_col = 0
        self.best_threshold = 0
        self.best_option = 1

    def fit(self,x,y,sample_weight = None):
        if sample_weight is None:
            sample_weight = np.ones(len(x)) / len(x) #初始化权重矩阵
        n = x.shape[1]
        for i in range(n):
            feature = x[:, i] #特征列
            #print(feature)
            feature_sort = np.sort(np.unique(feature)) #从小到大排序
            #print(feature_sort)
            for j in range(len(feature_sort) - 1):
                threshold = (feature_sort[j] + feature_sort[j+1]) / 2 #逐一设定该特征列阈值
                for option in (0,1):
                    if option == 1:
                        y_pred = 2 * (feature >= threshold) - 1
                    else:
                        y_pred = 2 * (feature < threshold) - 1
                    #print(y_pred)
                    #print(sample_weight)
                    #print(y)
                    error = np.sum((y_pred != y) * sample_weight)
                    if error < self.best_error:
                        self.best_error = error
                        self.best_option = option 
                        self.best_feature_col = i
                        self.best_threshold = threshold
        return self
    
    def predict(self,x):
        feature = x[:,self.best_feature_col]
        if self.best_option == 1:
            pred = 2 * (feature >= self.best_threshold) - 1
        else :
            pred = 2 * (feature < self.best_threshold) - 1
        return pred
    
    def score(self,x,y,sample_weight = None):
        pred = self.predict(x)
        if sample_weight is not None:
            return np.sum((pred == y) * sample_weight)
        return np.mean((pred == y))


class Adaboost():
    def __init__(self,n_learners = 5):
        self.n_learners = n_learners
        self.learners = []
        self.alphas = []
        self.num_learner = 0
    
    def fit(self,x,y,xtest,ytest):
        auc_all = []
        scores = []
        start = time.perf_counter()
        sample_weight = np.ones(len(x)) / len(x)
        for _ in range(self.n_learners):
            starti = time.perf_counter()
            ctf = TreeClassifier_weight().fit(x,y,sample_weight)
            alpha = 1/2 * np.log((1-ctf.best_error)/ctf.best_error)
            y_pred = ctf.predict(x)
            #print(y_pred)
            sample_weight *= np.exp(- alpha * y_pred * y)
            sample_weight /= np.sum(sample_weight)
            self.learners.append(ctf)
            self.alphas.append(alpha)
            self.num_learner += 1 #已经运行的基学习器数量

            y_pred_ada = self.auc(xtest)
            #计算fpr(假阳性率),tpr(真阳性率),thresholds(阈值)[绘制ROC曲线要用到这几个值]
            fpr,tpr,threshold=roc_curve(ytest ,y_pred_ada ,pos_label = 1)
            #计算auc
            roc_auc=auc(fpr,tpr)
            auc_all.append(roc_auc)
            
            score1 = self.score_every(xtest, ytest)
            scores.append(score1)

            print("第%d个基学习器运行时间: %f s，AUC分数为：%f，准确度为：%f" % (_+1,time.perf_counter() - starti,roc_auc,score1))

        with open (file = 'BoostMain.csv',mode = 'w+',encoding='utf-8') as file:
            for i in range(self.num_learner):
                file.write(str(i+1)+','+str(auc_all[i])+',')
                #file.write(str(auc_all[i]),end=' ')
                file.write(str(scores[i]))
                file.write('\n')
        print('写入完成！')

        print("adaboost运行时间: %f s" % (time.perf_counter() - start))

        return self
    
    def auc(self, x):
        y_pred = np.empty((len(x), self.num_learner))  
        for i in range(self.num_learner):
            y_pred[:, i] = self.learners[i].predict(x)
        y_pred = y_pred * np.array(self.alphas)  
        return np.sum(y_pred, axis=1)

    def predict_every(self, x):
        y_pred = np.empty((len(x), self.num_learner))  
        for i in range(self.num_learner):
            y_pred[:, i] = self.learners[i].predict(x)
        y_pred = y_pred * np.array(self.alphas)  
        return 2*(np.sum(y_pred, axis=1)>0)-1

    def predict(self, x):
        y_pred = np.empty((len(x), self.n_learners))  
        for i in range(self.n_learners):
            y_pred[:, i] = self.learners[i].predict(x)
        y_pred = y_pred * np.array(self.alphas)  
        return 2*(np.sum(y_pred, axis=1)>0)-1

    def score_every(self, x, y):
        y_pred = self.predict_every(x)
        return np.mean(y_pred==y)

    def score(self, x, y):
        y_pred = self.predict(x)
        return np.mean(y_pred==y)

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
    #print(ytrain)
    #score = TreeClassifier_weight().fit(xtrain,ytrain).score(xtest,ytest)
    #print(score)
    #build_tree(xtrain,ytrain,xtest,ytest)
    Adaboost().fit(xtrain, ytrain, xtest, ytest)
    plot_auc()
main()