import numpy as np
import math
# sigmoid
def sigmoid(x):
    # （需要填写的地方，输入x返回sigmoid(x)）
    return 1.0/(1+np.exp(-x))


def deriv_sigmoid(x):
    # （需要填写的地方，输入x返回sigmoid(x)在x点的梯度）
    return sigmoid(x)*(1-sigmoid(x))

# loss
def mse_loss(self, y_true, y_pred):
    # （需要填写的地方，输入真实标记和预测值返回他们的MSE（均方误差）,其中真实标记和预测值都是长度相同的向量）
    '''
    #正则项
    if len(y_true) == len(y_pred):
        lamb = 0.1
        return lamb * (sum([(x - y) ** 2 / 2 for x, y in zip(y_true, y_pred)]) / len(y_true)) + ( 1 - lamb) * ((self.b1 + self.b2 + self.b3 +self.w1 + self.w2 + self.w3+ self.w4 + self.w5 + self.w6) ** 2)
    else:
        return None
    #交叉熵
    if len(y_true) == len(y_pred):
        delta = 1e-7
        return -(sum(x * np.log(y + delta) for x, y in zip(y_true, y_pred))) / len(y_true)
    else:
        return None
    '''
    if len(y_true) == len(y_pred):
        return (sum([(x - y) ** 2 / 2 for x, y in zip(y_true, y_pred)]) / len(y_true)) 
    else:
        return None

class NeuralNetwork_221():
    def __init__(self):
        # weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        self.w7 = np.random.normal()
        self.w8 = np.random.normal()
        self.w9 = np.random.normal()
        self.w10 = np.random.normal()
        # biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        self.b4 = np.random.normal()
        self.b5 = np.random.normal()
        # 以上为神经网络中的变量，其中具体含义见网络图
    
    #adam法
    def adam(self ,data ,all_y_trues):
        # 初始化
        learn_rate = 0.01  # 学习率
        epochs = 500  # 迭代次数

        beta_1 = 0.9  # 算法作者建议的默认值
        beta_2 = 0.999  # 算法作者建议的默认值
        eps = 0.00000001  #算法作者建议的默认值
        mt = [0,0,0,0,0,0,0,0,0] #因为有9个参数
        vt = [0,0,0,0,0,0,0,0,0]
        #参数列表
        theta=[]
        for i in range(0,10):
            theta.append(np.random.normal())

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                sum_h1 = x[0] * self.w1 + x[1] * self.w2 # （需要填写的地方，含义为隐层第一个节点收到的输入之和）
                h1 = sigmoid(sum_h1 + self.b1) # （需要填写的地方，含义为隐层第一个节点的输出）

                sum_h2 = x[0] * self.w3 + x[1] * self.w4 # （需要填写的地方，含义为隐层第二个节点收到的输入之和）
                h2 = sigmoid(sum_h2 + self.b2) # （需要填写的地方，含义为隐层第二个节点的输出）

                sum_ol =  h1 * self.w5 + h2 * self.w6 # （需要填写的地方，含义为输出层节点收到的输入之和）
                ol = sigmoid(sum_ol + self.b3) # （需要填写的地方，含义为输出层节点的对率输出）
                y_pred = ol

                # 以下部分为计算梯度，请完成
                d_L_d_ypred = y_pred - y_true # （需要填写的地方，含义为损失函数对输出层对率输出的梯度）
                # 输出层梯度
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_ol + self.b3) # （需要填写的地方，含义为输出层对率输出对w5的梯度）
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_ol + self.b3) # （需要填写的地方，含义为输出层对率输出对w6的梯度）
                d_ypred_d_b3 = deriv_sigmoid(sum_ol + self.b3) # （需要填写的地方，含义为输出层对率输出对b3的梯度）
                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_ol + self.b3) # （需要填写的地方，含义为输出层输出对率对隐层第一个节点的输出的梯度）
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_ol + self.b3) # （需要填写的地方，含义为输出层输出对率对隐层第二个节点的输出的梯度）

                # 隐层梯度
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1 + self.b1) # （需要填写的地方，含义为隐层第一个节点的输出对w1的梯度）
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1 + self.b1) # （需要填写的地方，含义为隐层第一个节点的输出对w2的梯度）
                d_h1_d_b1 = deriv_sigmoid(sum_h1 + self.b1) # （需要填写的地方，含义为隐层第一个节点的输出对b1的梯度）

                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2 + self.b2) # （需要填写的地方，含义为隐层第二个节点的输出对w3的梯度）
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2 + self.b2) # （需要填写的地方，含义为隐层第二个节点的输出对w4的梯度）
                d_h2_d_b2 = deriv_sigmoid(sum_h2 + self.b2) # （需要填写的地方，含义为隐层第二个节点的输出对b2的梯度）

                # 更新权重和偏置
                gradient = [] #参数列表（顺序为w1,w2,w3,w4,w5,w6,b1,b2,b3）
                gradient.append(d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1)
                gradient.append(d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2)
                gradient.append(d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3)
                gradient.append(d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4)
                gradient.append(d_L_d_ypred * d_ypred_d_w5)
                gradient.append(d_L_d_ypred * d_ypred_d_w6)
                gradient.append(d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1)
                gradient.append(d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2)
                gradient.append(d_L_d_ypred * d_ypred_d_b3)
                
                for i in range(0,9):
                    mt[i] = beta_1 * mt[i] + (1 - beta_1) * gradient[i]
                    vt[i] = beta_2 * vt[i] + (1 - beta_2) * (gradient[i]**2)
                    mtt = mt[i] / (1 - (beta_1**(epoch + 1)))
                    vtt = vt[i] / (1 - (beta_2**(epoch + 1)))
                    vtt_sqrt = math.sqrt(vtt)
                    theta[i] = theta[i] - learn_rate * mtt / (vtt_sqrt + eps)
        
            # 计算epoch的loss
            if epoch % 10 == 0:
                #参数赋值
                self.w1 = theta[0]
                self.w2 = theta[1]
                self.w3 = theta[2]
                self.w4 = theta[3]
                self.w5 = theta[4]
                self.w6 = theta[5]
                self.b1 = theta[6]
                self.b2 = theta[7]
                self.b3 = theta[8]
                y_preds = np.apply_along_axis(self.predict, 1, data)
                loss = mse_loss(self, all_y_trues, y_preds)
                print("Epoch %d loss: %.3f", (epoch, loss))

    def predict(self,x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        h3 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        h4 = sigmoid(self.w7 * h1 + self.w8 * h2 + self.b4)
        o1 = sigmoid(self.w9 * h3 + self.w10 * h4 + self.b5)
        return o1

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 500
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                '''
                #单隐层
                # 以下部分为向前传播过程，请完成
                sum_h1 = x[0] * self.w1 + x[1] * self.w2 # （需要填写的地方，含义为隐层第一个节点收到的输入之和）
                h1 = sigmoid(sum_h1 + self.b1) # （需要填写的地方，含义为隐层第一个节点的输出）

                sum_h2 = x[0] * self.w3 + x[1] * self.w4 # （需要填写的地方，含义为隐层第二个节点收到的输入之和）
                h2 = sigmoid(sum_h2 + self.b2) # （需要填写的地方，含义为隐层第二个节点的输出）

                sum_ol =  h1 * self.w5 + h2 * self.w6 # （需要填写的地方，含义为输出层节点收到的输入之和）
                ol = sigmoid(sum_ol + self.b3) # （需要填写的地方，含义为输出层节点的对率输出）
                y_pred = ol

                # 以下部分为计算梯度，请完成
                d_L_d_ypred = y_pred - y_true # （需要填写的地方，含义为损失函数对输出层对率输出的梯度）
                # 输出层梯度
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_ol + self.b3) # （需要填写的地方，含义为输出层对率输出对w5的梯度）
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_ol + self.b3) # （需要填写的地方，含义为输出层对率输出对w6的梯度）
                d_ypred_d_b3 = deriv_sigmoid(sum_ol + self.b3) # （需要填写的地方，含义为输出层对率输出对b3的梯度）
                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_ol + self.b3) # （需要填写的地方，含义为输出层输出对率对隐层第一个节点的输出的梯度）
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_ol + self.b3) # （需要填写的地方，含义为输出层输出对率对隐层第二个节点的输出的梯度）

                # 隐层梯度
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1 + self.b1) # （需要填写的地方，含义为隐层第一个节点的输出对w1的梯度）
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1 + self.b1) # （需要填写的地方，含义为隐层第一个节点的输出对w2的梯度）
                d_h1_d_b1 = deriv_sigmoid(sum_h1 + self.b1) # （需要填写的地方，含义为隐层第一个节点的输出对b1的梯度）

                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2 + self.b2) # （需要填写的地方，含义为隐层第二个节点的输出对w3的梯度）
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2 + self.b2) # （需要填写的地方，含义为隐层第二个节点的输出对w4的梯度）
                d_h2_d_b2 = deriv_sigmoid(sum_h2 + self.b2) # （需要填写的地方，含义为隐层第二个节点的输出对b2的梯度）

                # 更新权重和偏置
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5 # （需要填写的地方，更新w5）
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6 # （需要填写的地方，更新w6）
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3 # （需要填写的地方，更新b3）

                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1  # （需要填写的地方，更新w1）
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2 # （需要填写的地方，更新w2）
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1 # （需要填写的地方，更新b1）

                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3 # （需要填写的地方，更新w3）
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4  # （需要填写的地方，更新w4）
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2  # （需要填写的地方，更新b2）

                '''
                #两层隐层
                # 以下部分为向前传播过程，请完成
                sum_h1 = x[0] * self.w1 + x[1] * self.w2 # （需要填写的地方，含义为隐层第一个节点收到的输入之和）
                h1 = sigmoid(sum_h1 + self.b1) # （需要填写的地方，含义为隐层第一个节点的输出）

                sum_h2 = x[0] * self.w3 + x[1] * self.w4 # （需要填写的地方，含义为隐层第二个节点收到的输入之和）
                h2 = sigmoid(sum_h2 + self.b2) # （需要填写的地方，含义为隐层第二个节点的输出）

                sum_h3 = h1 * self.w5 + h2 * self.w6
                h3 = sigmoid(sum_h3 + self.b3)

                sum_h4 = h1 * self.w7 + h2 * self.w8
                h4 = sigmoid(sum_h4 + self.b4)

                sum_ol =  h3 * self.w9 + h4 * self.w10 # （需要填写的地方，含义为输出层节点收到的输入之和）
                ol = sigmoid(sum_ol + self.b5) # （需要填写的地方，含义为输出层节点的对率输出）
                y_pred = ol

                # 以下部分为计算梯度，请完成
                d_L_d_ypred = y_pred - y_true # （需要填写的地方，含义为损失函数对输出层对率输出的梯度）
                # 输出层梯度
                d_ypred_d_w9 = h3 * deriv_sigmoid(sum_ol + self.b5)
                d_ypred_d_w10 = h4 * deriv_sigmoid(sum_ol + self.b5)
                d_ypred_d_b5 = deriv_sigmoid(sum_ol + self.b5)
                d_ypred_d_h3 = self.w9 * deriv_sigmoid(sum_ol + self.b5)
                d_ypred_d_h4 = self.w10 * deriv_sigmoid(sum_ol + self.b5)

                # 第二层梯度
                d_h3_d_w5 = h1 * deriv_sigmoid(sum_h3 + self.b3) # （需要填写的地方，含义为输出层对率输出对w5的梯度）
                d_h3_d_w6 = h2 * deriv_sigmoid(sum_h3 + self.b3) # （需要填写的地方，含义为输出层对率输出对w6的梯度）
                d_h3_d_b3 = deriv_sigmoid(sum_h3 + self.b3)
                d_h3_d_h1 = self.w5 * deriv_sigmoid(sum_h3 + self.b3) 
                d_h3_d_h2 = self.w6 * deriv_sigmoid(sum_h3 + self.b3) 

                d_h4_d_w7 = h1 * deriv_sigmoid(sum_h4 + self.b4)
                d_h4_d_w8 = h2 * deriv_sigmoid(sum_h4 + self.b4)
                d_h4_d_b4 = deriv_sigmoid(sum_h4 + self.b4)
                d_h4_d_h1 = self.w7 * deriv_sigmoid(sum_h4 + self.b4)# （需要填写的地方，含义为输出层输出对率对隐层第一个节点的输出的梯度）
                d_h4_d_h2 = self.w8 * deriv_sigmoid(sum_h4 + self.b4)# （需要填写的地方，含义为输出层输出对率对隐层第二个节点的输出的梯度）

                # 第一层梯度
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1 + self.b1) # （需要填写的地方，含义为隐层第一个节点的输出对w1的梯度）
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1 + self.b1) # （需要填写的地方，含义为隐层第一个节点的输出对w2的梯度）
                d_h1_d_b1 = deriv_sigmoid(sum_h1 + self.b1) # （需要填写的地方，含义为隐层第一个节点的输出对b1的梯度）

                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2 + self.b2) # （需要填写的地方，含义为隐层第二个节点的输出对w3的梯度）
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2 + self.b2) # （需要填写的地方，含义为隐层第二个节点的输出对w4的梯度）
                d_h2_d_b2 = deriv_sigmoid(sum_h2 + self.b2) # （需要填写的地方，含义为隐层第二个节点的输出对b2的梯度）

                # 更新权重和偏置
                self.w9 -= learn_rate * d_L_d_ypred * d_ypred_d_w9
                self.w10 -= learn_rate * d_L_d_ypred * d_ypred_d_w10
                self.b5 -= learn_rate * d_L_d_ypred * d_ypred_d_b5

                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w5 # （需要填写的地方，更新w5）
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_w6 # （需要填写的地方，更新w6）
                self.w7 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h4_d_w7
                self.w8 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h4_d_w8
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_h3 * d_h3_d_b3 # （需要填写的地方，更新b3）
                self.b4 -= learn_rate * d_L_d_ypred * d_ypred_d_h4 * d_h4_d_b4

                self.w1 -= learn_rate * d_L_d_ypred * (d_ypred_d_h3 * d_h3_d_h1 * d_h1_d_w1 + d_ypred_d_h4 * d_h4_d_h1 * d_h1_d_w1)  # （需要填写的地方，更新w1）
                self.w2 -= learn_rate * d_L_d_ypred * (d_ypred_d_h3 * d_h3_d_h1 * d_h1_d_w2 + d_ypred_d_h4 * d_h4_d_h1 * d_h1_d_w2) # （需要填写的地方，更新w2）
                self.b1 -= learn_rate * d_L_d_ypred * (d_ypred_d_h3 * d_h3_d_h1 * d_h1_d_b1 + d_ypred_d_h4 * d_h4_d_h1 * d_h1_d_b1) # （需要填写的地方，更新b1）

                self.w3 -= learn_rate * d_L_d_ypred * (d_ypred_d_h3 * d_h3_d_h2 * d_h2_d_w3 + d_ypred_d_h4 * d_h4_d_h2 * d_h2_d_w3) # （需要填写的地方，更新w3）
                self.w4 -= learn_rate * d_L_d_ypred * (d_ypred_d_h3 * d_h3_d_h2 * d_h2_d_w4 + d_ypred_d_h4 * d_h4_d_h2 * d_h2_d_w4)  # （需要填写的地方，更新w4）
                self.b2 -= learn_rate * d_L_d_ypred * (d_ypred_d_h3 * d_h3_d_h2 * d_h2_d_b2 + d_ypred_d_h4 * d_h4_d_h2 * d_h2_d_b2)  # （需要填写的地方，更新b2）
                

            # 计算epoch的loss
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.predict, 1, data)
                loss = mse_loss(self, all_y_trues, y_preds)
                print("Epoch %d loss: %.3f", (epoch, loss))
def main():
    import numpy as np
    X_train = np.genfromtxt('./data/train_feature.csv', delimiter=',')
    y_train = np.genfromtxt('./data/train_target.csv', delimiter=',')
    X_test = np.genfromtxt('./data/test_feature.csv', delimiter=',')#读取测试样本特征
    network = NeuralNetwork_221()
    network.train(X_train, y_train)
    y_pred=[]
    for i in X_test:
        y_pred.append(network.predict(i))#将预测值存入y_pred(list)内
    ##############
    # （需要填写的地方，选定阈值，将输出对率结果转化为预测结果并输出）
    with open (file = '162050127_ypred.csv' , mode= 'w' , encoding= 'utf-8') as file:
        for i in y_pred:
            if i >= 0.5:
                file.write('1')
                file.write('\n')
            else:
                file.write('0')
                file.write('\n')
    ##############
    '''
    #测试训练集准确度
    y_pred_train=[]
    for i in X_train:
        if network.predict(i) >= 0.5:
            y_pred_train.append("1\n")
        else:
            y_pred_train.append("0\n")

    y_train=[]
    with open (file = 'D:\大学\机器学习\作业\第四章_神经网络\data\\train_target.csv' , mode= 'r' , encoding= 'utf-8') as file:
        for line in file:
            y_train.append(line)

    sum=0
    for i,j in zip(y_pred_train,y_train):
        if i==j:
            sum += 1
    print("训练集精度：",sum/400)
    '''
main()