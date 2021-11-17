import tensorflow
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout
from tensorflow.keras.layers import ELU, ReLU
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adamax, Nadam
from sklearn.model_selection import train_test_split

import numpy as np

import sys
sys.path.append('../utils/')
import get_HIN_graph
from logs import Logger

import pandas as pd 


def get_all_matrix():
    # 获取其具体索引值
    uniqueIDAPI, uniqueIDPackage, uniqueIDPermission = get_HIN_graph.getIDofAPIandPackage()
    uniqueIDAPP, labels = get_HIN_graph.getIDofAPP('../data/label.txt')

    # 获得API与package关系矩阵，以P简称表示
    matrixAPIPac = get_HIN_graph.CreateAPIPac_Matrix(uniqueIDAPI, uniqueIDPackage)

    # 获得APP与API关系矩阵，以A简称表示
    matrixAPPAPI = get_HIN_graph.CreateAPPAPI_Matrix(uniqueIDAPP, uniqueIDAPI, '../data/matrix/APK_API.txt')

    # 获得API与Permission之间的关系（PScout），以I简称表示
    matrixAPIPer = get_HIN_graph.CreateAPIPer_Matrix(uniqueIDAPI, uniqueIDPermission)

    # 获得APP与Hardwares之间的关系矩阵，以H简称表示
    # 获得APP与Permission之间的关系矩阵，以Q简称表示
    uniqueIDHard = {}
    uniqueIDHard, matrixAPPHard, matrixAPPPer = get_HIN_graph.CreateAPPHard_APPPer_Matrix(uniqueIDAPP, uniqueIDPermission, uniqueIDHard, '../data/matrix/APK_Per_hard.txt')
    return matrixAPIPac, matrixAPPAPI, matrixAPIPer, matrixAPPHard, matrixAPPPer, labels


def get_data(ker, labels):
    df = pd.DataFrame(ker)
    # print(df)
    # print(len(labels))
    # print(len(df))
    df['classification_id'] = labels.keys()
    df['classification_id'] = df['classification_id'].apply(lambda x: 1 if 'malware' in x else 0)

    feature_cols = df.iloc[:, :-1]
    x = feature_cols
    y = df['classification_id']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_train = x_train[:,:,np.newaxis]
    x_test = x_test[:,:,np.newaxis]
    # 将分类数据转换为独热编码
    # y_train = utils.to_categorical(y_train, 2)
    # y_test = utils.to_categorical(y_test, 2)
    y_train = utils.to_categorical(y_train, 2)
    y_test = utils.to_categorical(y_test, 2)
    # print(len(y_train))
    # num = len(y_train)

    print("训练数据数量:", x_train.shape[0])
    print("测试数据数量:", x_test.shape[0])
    # return x_train, x_test, y_train, y_test
    return x_train, x_test, y_train, y_test

def createCNN(input_shape, category):
    # 设置学习率
    LR = 0.004
    # 优化器
    OPTIMIZER = Nadam(learning_rate = LR, beta_1 = 0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    model = Sequential()

    # 建立模型
    # 第一层卷积
    model.add(Conv1D(filters=32, kernel_size=8, input_shape=input_shape, strides=1, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(ELU(alpha=1.0))
    model.add(Conv1D(filters=32, kernel_size=8, strides=1, padding='same'))
    model.add(BatchNormalization(axis=-1))
    model.add(ELU(alpha=1.0))
    model.add(MaxPooling1D(pool_size=8, strides=4, padding='same'))
    model.add(Dropout(0.1))

    # 全链接层
    model.add(Flatten())
    model.add(Dense(512, kernel_initializer=glorot_uniform(seed=0)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.7))
    model.add(Dense(512, kernel_initializer=glorot_uniform(seed=0)))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.7))

    # 预测
    model.add(Dense(category, kernel_initializer=glorot_uniform(seed=0)))
    model.add(Activation('softmax'))

    # 模型编译
    model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=["accuracy"])

    return model

class Animator(keras.callbacks.Callback):
    # 在训练开始前调用
    def on_train_begin(self, logs=None):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []
    
    # 在每一次训练结束时调用
    def on_epoch_end(self, epoch, logs=None):
        
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_accuracy'))
        
        '''
        with open('train2AAT.txt', 'a') as f:
            f.write(str(epoch) + '\n')
            f.write(str(self.logs[-1]) + '\n' + str(self.losses[-1]) + '\n' + str(self.acc[-1]) + '\n' + str(self.val_losses[-1]) + '\n' + str(self.val_acc[-1]) + '\n')
        '''

def trainCNN():
    # ker_AIIPPIIATrans, labels = test_AIIPPIIA()
    ker_AATrans, labels = test_AAT()
    # ker_QQTrans, labels = test_QQT()
    ker_HHTrans, labels = test_HHT()
    # ker_APPA, labels = test_APPA()
    # ker_APPIIPPATrans, labels = test_APPIIPPA()
    # ker_AIIATrans, labels = test_AIIA()

    test_new = []
    for i in range(ker_AATrans.shape[0]):
        test_new.append([])
        test_new[i] = ker_AATrans[i] * 0.5 + ker_HHTrans[i] * 0.5
    test_new = np.array(test_new)


    x_train, x_test, y_train, y_test = get_data(test_new, labels)
    
    '''
    train_animate = Animator()
    ker_AATrans, labels = test_QQT()
    x_train, x_test, y_train, y_test = get_data(ker_AATrans, labels)
    model = createCNN((6679, 1), 2)

    history = model.fit(x_train, y_train, batch_size=256, epochs=10, callbacks=[train_animate])

    # model.save('../data/Model/QQT_model.h5')

    # test
    score_test = model.evaluate(x_test, y_test, verbose=2)
    print("测试集精确度:", score_test[1])
    '''
    train_animate = Animator()
    model = createCNN((ker_AATrans.shape[0], 1), 2)
    model.fit(x_train, y_train, epochs=20, batch_size = 128)
    model.summary()

    # model.save("../data/Model/attention_1009.h5")

    score_test = model.evaluate(x_test, y_test, verbose=2)
    print("测试集精确度:", score_test[1])
    y_pred = model.predict(x_test, verbose=1)
    malware_True = 0 
    malware_False = 0
    benign_True = 0
    benign_False = 0
    for i in range(y_pred.shape[0]):
        if y_pred[i][0] > y_pred[i][1]:
            if y_test[i][0] == 1:
                benign_True += 1
            else:
                benign_False += 1
        else:
            if y_test[i][1] == 1:
                malware_True += 1
            else:
                malware_False += 1
    print('malware_True(TP):' + str(malware_True))
    print('malware_False(FP):' + str(malware_False))
    print('benign_True(TN):' + str(benign_True))
    print('benign_False(FN):' + str(benign_False))
    precision = malware_True/(malware_True + malware_False)
    print('Precision:' + str(precision))
    recall = malware_True/(malware_True + benign_False)
    print('Recall:' + str(recall))
    acc = (malware_True + benign_True)/(malware_True + malware_False + benign_True + benign_False)
    print('ACC:' + str(acc))
    print('F1:' + str(2 * precision * recall / (precision + recall)))
    # print(y_pred[0])

    # 绘制ROC曲线
    fpr, tpr, threshold = roc_curve(y_test[1], y_pred[1])
    roc_auc = auc(fpr, tpr)

    lw = 2
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

# 对矩阵进行转置操作
def trans(Matrix):
    return Matrix.transpose()

def getAATrans(matrixA, matrixATrans):
    return matrixA.dot(matrixATrans)

def test_AAT():
    # 对AA^T进行分类测试，表示每个APP调用的API对分类的影响
    P, A, I, H, Q, labels = get_all_matrix()
    AATrans = getAATrans(A, trans(A))
    # AATrans_test = getAATrans(A, trans(A_test))
    ker_AATrans = AATrans.toarray()
    return ker_AATrans, labels

def test_QQT():
    # 对QQ^T进行分类测试，表示每个APP所需求的权限对分类的影响
    P, A, I, H, Q, labels = get_all_matrix()
    QQTrans = getAATrans(Q, trans(Q))
    ker_QQTrans = QQTrans.toarray()
    return ker_QQTrans, labels
    

def test_HHT():
    # 对HH^T进行分类测试，表示APP调用了相同的权限信息
    P, A, I, H, Q, labels = get_all_matrix()
    HHTrans = getAATrans(H, trans(H))
    ker_HHTrans = HHTrans.toarray()
    return ker_HHTrans, labels

def test_APPA():
    # 对路径APPA进行分类测试，表示两个APP调用的API属于同一个包
    P, A, I, H, Q, labels = get_all_matrix()
    AP = A.dot(P)
    APP = AP.dot(trans(P))
    APPA = APP.dot(trans(A))
    ker_APPA = APPA.toarray()
    return ker_APPA, labels

def test_APPIIPPA():
    # 路径APPIIPPA表示调用的API中属于同一个包的具有相同的权限
    P, A, I, H, Q, labels = get_all_matrix()
    APPIIPPA = A.dot(P).dot(trans(P)).dot(I).dot(trans(I)).dot(P).dot(trans(P)).dot(trans(A))
    ker_APPIIPPA = APPIIPPA.toarray()
    return ker_APPIIPPA, labels

def test_AIIPPIIA():
    # 路径AIIPPIIA表示与调用的API具有相同权限的API属于同一个包
    P, A, I, H, Q, labels = get_all_matrix()
    AIIPPIIA = A.dot(I).dot(trans(I)).dot(P).dot(trans(P)).dot(I).dot(trans(I)).dot(trans(A))
    ker_AIIPPIIA = AIIPPIIA.toarray()
    return ker_AIIPPIIA, labels

if __name__ == '__main__':
    trainCNN()
