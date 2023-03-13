import numpy as np  
import matplotlib.pyplot as plt  

X_train = np.genfromtxt('./data/train_feature.csv', delimiter=',')
y_train = np.genfromtxt('./data/train_target.csv', delimiter=',')

fig = plt.figure()  
ax1 = fig.add_subplot(111)  
#设置标题  
ax1.set_title('Scatter Plot')  
#设置X轴标签  
plt.xlabel('X1')  
#设置Y轴标签  
plt.ylabel('X2')  
#画散点图 
for x,y in zip(X_train, y_train):
    if y == 1:
        s1=ax1.scatter(x[0],x[1],c = 'r',marker = 'o')  
    else:
        s2=ax1.scatter(x[0],x[1],c = 'b',marker = 'o')  
#设置图标  
plt.legend((s1,s2),('positive','negative'))  
#显示所画的图  
plt.show()