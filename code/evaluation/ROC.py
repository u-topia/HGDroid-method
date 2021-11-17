# 该文档用于绘制ROC曲线，主要针对使用深度学习算法方案进行绘制
import sys
sys.path.append('../')
from utils.logs import Logger
import utils.get_HIN_graph as get_HIN_graph

import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from tensorflow.keras import utils

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

def Get_data(ker, labels, k = 0, m = 0):
    if k == 0:
    	df = pd.DataFrame(ker)
    	# print(df)
    	# print(len(labels))
    	# print(len(df))
    	df['classification_id'] = labels.keys()
    	df['classification_id'] = df['classification_id'].apply(lambda x: 1 if 'malware' in x else 0)
	
    	feature_cols = df.iloc[:, :-1]
    	x = feature_cols
    	y = df['classification_id']
    	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 12)
    	# 将分类数据转换为独热编码
    	if m == 0:
    		y_train = utils.to_categorical(y_train, 2)
    		y_test = utils.to_categorical(y_test, 2)
    		y = utils.to_categorical(y, 2)
    		x_train = np.array(x_train)
    		# print(type(x_train))
    		# print(x_train.shape)
    		x_test = np.array(x_test)
    		x_train = x_train[:,:,np.newaxis]
    		x_test = x_test[:,:,np.newaxis]
    		# print(len(y_train))
    		# num = len(y_train)
    		x = np.array(x)
    		x = x[:,:,np.newaxis]
    		# print(x)
    else:
    	length = len(labels)
    	y = np.zeros(length)
    	for key in labels:
    		if 'malware' in key:
    			y[labels[key]] = 1
    		else:
    			y[labels[key]] = 0
    	x = ker
    	# print(x[0])
    	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 12)
    	x_train = np.array(x_train)
    	# print(type(x_train))
    	# print(x_train.shape)
    	x_test = np.array(x_test)
    	# print(x_test)
    	x_train = x_train[:,:,:,np.newaxis]
    	x_test = x_test[:,:,:,np.newaxis]
    	# 将分类数据转换为独热编码
    	y_train = utils.to_categorical(y_train, 2)
    	y_test = utils.to_categorical(y_test, 2)
    	# print(len(y_train))
    	# num = len(y_train)
    	x = np.array(x)
    	x = x[:,:,:,np.newaxis]
    	y = utils.to_categorical(y, 2)

    print("训练数据数量:", x_train.shape[0])
    print("测试数据数量:", x_test.shape[0])
    # return x_train, x_test, y_train, y_test
    return x_train, x_test, y_train, y_test, x, y

# 对矩阵进行转置操作
def trans(Matrix):
    return Matrix.transpose()

def getAATrans(matrixA, matrixATrans):
    return matrixA.dot(matrixATrans)

def test_AAT():
    # 对AA^T进行分类测试，表示每个APP调用的API对分类的影响
    P, A, I, H, Q, labels = get_all_matrix()
    AATrans = getAATrans(A, trans(A))
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

def test_AIIA():
	P, A, I, H, Q, labels = get_all_matrix()
	AI = A.dot(I)
	AII = AI.dot(trans(I))
	AIIA = AII.dot(trans(A))
	ker_AIIA = AIIA.toarray()
	return ker_AIIA, labels

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

def test_con():
    P, A, I, H, Q, labels = get_all_matrix()

    AATrans = getAATrans(A, trans(A))
    ker_AATrans = AATrans.toarray()

    QQTrans = getAATrans(Q, trans(Q))
    ker_QQTrans = QQTrans.toarray()

    HHTrans = getAATrans(H, trans(H))
    ker_HHTrans = HHTrans.toarray()

    AP = A.dot(P)
    APP = AP.dot(trans(P))
    APPA = APP.dot(trans(A))
    ker_APPA = APPA.toarray()

    AI = A.dot(I)
    AII = AI.dot(trans(I))
    AIIA = AII.dot(trans(A))
    ker_AIIA = AIIA.toarray()

    APPIIPPA = A.dot(P).dot(trans(P)).dot(I).dot(trans(I)).dot(P).dot(trans(P)).dot(trans(A))
    ker_APPIIPPA = APPIIPPA.toarray()

    AIIPPIIA = A.dot(I).dot(trans(I)).dot(P).dot(trans(P)).dot(I).dot(trans(I)).dot(trans(A))
    ker_AIIPPIIA = AIIPPIIA.toarray()

    return ker_AATrans, ker_QQTrans, ker_HHTrans, ker_APPA, ker_AIIA, ker_APPIIPPA, ker_AIIPPIIA, labels

from sklearn.naive_bayes import GaussianNB
from Arithmetic.DNNtrain import CreateDNNModel
from Arithmetic import CNNtrain
def draw_ROC():
	ker_AATrans, labels = test_AAT()
	ker_HHTrans, labels = test_HHT()

	test_new = []
	for i in range(ker_AATrans.shape[0]):
		test_new.append([])
		test_new[i] = ker_AATrans[i] * 0.5 + ker_HHTrans[i] * 0.5
	test_new = np.array(test_new)

	x_train, x_test, y_train, y_test, x, y = Get_data(test_new, labels, m = 1)
	
	# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 12)

	# 加入SVM的绘图点测试
	clf = LinearSVC(random_state=0, tol = 1e-5)
	fitted = clf.fit(x_train, y_train)
	pred = fitted.predict(x_test)
	clf.score(x_test, y_test)
	accuracy = fitted.score(x_test, y_test)
	print('accuracy of SVM is ' + str(accuracy))
	fpr_svm, tpr_svm, threshold_svm = roc_curve(y_test, pred)
	roc_auc_svm = auc(fpr_svm, tpr_svm)

	# 加入NB的绘图点测试
	clf1 = GaussianNB()
	fitted1 = clf1.fit(x_train, y_train)
	pred1 = fitted1.predict(x_test)
	clf1.score(x_test, y_test)
	accuracy = fitted1.score(x_test, y_test)
	print('accuracy of NB is ' + str(accuracy))
	fpr_nb, tpr_nb, threshold_nb = roc_curve(y_test, pred1)
	roc_auc_nb = auc(fpr_nb, tpr_nb)

	x_train, x_test, y_train, y_test, x, y = Get_data(test_new, labels)
	# 加入DNN的绘图点测试
	Input_shape = (ker_AATrans.shape[0], 1)
	model = CreateDNNModel(Input_shape)

	model.fit(x_train, y_train, epochs = 10, batch_size = 128)
	score_test = model.evaluate(x_test, y_test, verbose=2)
	print("dnn测试集精确度:", score_test[1])
	y_pred = model.predict(x_test, verbose=1)
	fpr_dnn, tpr_dnn, threshold_dnn = roc_curve(y_test[1], y_pred[1])
	roc_auc_dnn = auc(fpr_dnn, tpr_dnn)

	# 加入CNN的绘图点测试
	train_animate = CNNtrain.Animator()
	model1 = CNNtrain.createCNN((ker_AATrans.shape[1], 1), 2)
	model1.fit(x_train, y_train, epochs = 8, batch_size = 128)
	score_test = model1.evaluate(x_test, y_test, verbose=2)
	print("cnn测试集精确度:", score_test[1])
	y_pred = model.predict(x_test, verbose=1)
	fpr_cnn, tpr_cnn, threshold_cnn = roc_curve(y_test[1], y_pred[1])
	roc_auc_cnn = auc(fpr_cnn, tpr_cnn)

	# 绘制图像
	lw = 2
	plt.figure(figsize=(8, 5))
	plt.plot(fpr_svm, tpr_svm, color='darkorange', linestyle=':',
	         lw=lw, label='SVM')  ###假正率为横坐标，真正率为纵坐标做曲线
	# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.plot(fpr_nb, tpr_nb, color='b', linestyle='-.', lw=lw, label = 'NB')
	plt.plot(fpr_dnn, tpr_dnn, color='g', linestyle='-', lw=lw, label = 'DNN')
	plt.plot(fpr_cnn, tpr_cnn, color='blueviolet', linestyle='--', lw=lw, label='CNN')
	plt.xlim([0.0, 0.03])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	# plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()


if __name__ == '__main__':
	draw_ROC()






