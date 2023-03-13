from sklearn import svm
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
def load_data(x_train,y_train):
    train_data = []
    train_feature = []
    for x,y in zip(x_train,y_train):
        x=list(x)
        line = []
        for j in range(2):
            line.append(x[j])
        train_data.append(line)
        train_feature.append(y)
    return train_data,train_feature

def gauss(xtrain,ytrain,xtest,ytest):
    gauss_svm = svm.SVC(C=0.5, kernel='rbf', class_weight='balanced')
    gauss_svm.fit(xtrain, ytrain)
    predict_gauss = []
    for i in xtest:
        i=np.array(i).reshape(1, -1)
        predict_gauss.append(float(gauss_svm.predict(i)))
    print("高斯核数据集的准确率：", gauss_svm.score(xtrain, ytrain))
    print('高斯核验证集的准确率: %s' % (accuracy_score(y_pred=predict_gauss, y_true=ytest)))

    n_Support_vector = gauss_svm.n_support_#支持向量个数
    print("支持向量个数为： ",n_Support_vector)
    Support_vector_index = gauss_svm.support_#支持向量索引

    ###原始数据的绘制（此时得到图1）
    for x,y in zip(xtrain,ytrain):
        if y == 1:
            plt.scatter(x[0],x[1],color='red')             #绘制出类别0和类别1
        else:
            plt.scatter(x[0],x[1],color='blue')      #绘制出类别0和类别1
    plt.title("Gauss")
    

    for j in Support_vector_index:
        if ytrain[j] == 1: 
            plt.scatter(xtrain[j][0],xtrain[j][1], color='red',s=100, alpha=0.5, linewidth=1.5, edgecolor='red')
        if ytrain[j] == 0: 
            plt.scatter(xtrain[j][0],xtrain[j][1], color='blue',s=100, alpha=0.5, linewidth=1.5, edgecolor='blue')

    plt.show()

def linear(xtrain,ytrain,xtest,ytest):

    linear_svm = svm.SVC(C=0.5 ,class_weight='balanced',kernel='linear')
    linear_svm.fit(xtrain,ytrain)
    predict_linear = []
    for i in xtest:
        i=np.array(i).reshape(1, -1)
        predict_linear.append(float(linear_svm.predict(i)))
    print("线性核数据集的准确率：", linear_svm.score(xtrain, ytrain))
    print('线性核验证集的准确率为：{}'.format(accuracy_score(y_pred=predict_linear, y_true=ytest)))

    n_Support_vector = linear_svm.n_support_#支持向量个数
    print("支持向量个数为： ",n_Support_vector)
    Support_vector_index = linear_svm.support_#支持向量索引
    w = linear_svm.coef_#方向向量W
    b = linear_svm.intercept_#截距项b
    
    ###原始数据的绘制（此时得到图1）
    for x,y in zip(xtrain,ytrain):
        if y == 1:
            plt.scatter(x[0],x[1],color='red')             #绘制出类别0和类别1
        else:
            plt.scatter(x[0],x[1],color='blue')      #绘制出类别0和类别1
    plt.title("Linear")

    for j in Support_vector_index:#绘制支持向量
        if ytrain[j] == 1: 
            plt.scatter(xtrain[j][0],xtrain[j][1], color='red',s=100, alpha=0.5, linewidth=1.5, edgecolor='red')
        if ytrain[j] == 0: 
            plt.scatter(xtrain[j][0],xtrain[j][1], color='blue',s=100, alpha=0.5, linewidth=1.5, edgecolor='blue')
    
    #绘制超平面
    x = np.arange(5,12,0.01)
    y = (w[0][0]*x+b)/(-1*w[0][1])
    plt.scatter(x,y,s=5,marker = 'h')

    plt.show()

def main():
    #导入数据
    X_train = np.genfromtxt('./data/train_feature.csv', delimiter=',')
    Y_train = np.genfromtxt('./data/train_target.csv', delimiter=',')
    train_data = []
    train_feature = []
    train_data,train_feature=load_data(X_train,Y_train)
    #划分训练集和验证集
    xtrain,xtest,ytrain,ytest = train_test_split(train_data,train_feature,test_size=0.2)


    #线性核处理
    linear(xtrain,ytrain,xtest,ytest)

    #高斯核处理
    gauss(xtrain,ytrain,xtest,ytest)

main()