# from sklearn.model_selection import train_test_split
from tensorflow.keras import utils
from keras.models import Sequential, Model
from tensorflow.keras.layers import ReLU, PReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout, Input
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adamax, Nadam

# from DNN_Attention import Attention, Average, WeightedSum, WeightedAspectEmb, MaxMargin

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import numpy as np 
import tensorflow as tf
import pandas as pd

import sys
sys.path.append('../utils/')
import get_HIN_graph
from logs import Logger


def get_all_matrix():
    # 获取其具体索引值
    uniqueIDAPI, uniqueIDPackage, uniqueIDPermission = get_HIN_graph.getIDofAPIandPackage()
    uniqueIDAPP, labels = get_HIN_graph.getIDofAPP('../data/label.txt')
    # uniqueIDAPP, labels = get_HIN_graph.getIDofAPP('../data/AndroZoo/labels.txt')

    # 获得API与package关系矩阵，以P简称表示
    matrixAPIPac = get_HIN_graph.CreateAPIPac_Matrix(uniqueIDAPI, uniqueIDPackage)

    # 获得APP与API关系矩阵，以A简称表示
    matrixAPPAPI = get_HIN_graph.CreateAPPAPI_Matrix(uniqueIDAPP, uniqueIDAPI, '../data/matrix/APK_API.txt')
    # matrixAPPAPI = get_HIN_graph.CreateAPPAPI_Matrix(uniqueIDAPP, uniqueIDAPI, '../data/matrix/APK_API_test.txt')

    # 获得API与Permission之间的关系（PScout），以I简称表示
    matrixAPIPer = get_HIN_graph.CreateAPIPer_Matrix(uniqueIDAPI, uniqueIDPermission)

    # 获得APP与Hardwares之间的关系矩阵，以H简称表示
    # 获得APP与Permission之间的关系矩阵，以Q简称表示
    uniqueIDHard = {}
    uniqueIDHard, matrixAPPHard, matrixAPPPer = get_HIN_graph.CreateAPPHard_APPPer_Matrix(uniqueIDAPP, uniqueIDPermission, uniqueIDHard, '../data/matrix/APK_Per_hard.txt')
    # uniqueIDHard, matrixAPPHard, matrixAPPPer = get_HIN_graph.CreateAPPHard_APPPer_Matrix(uniqueIDAPP, uniqueIDPermission, uniqueIDHard, '../data/matrix/APK_Per_hard_test.txt')

    return matrixAPIPac, matrixAPPAPI, matrixAPIPer, matrixAPPHard, matrixAPPPer, labels

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


def Get_data(ker, labels, k = 0):
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
    	x_train = np.array(x_train)
    	# print(type(x_train))
    	# print(x_train.shape)
    	x_test = np.array(x_test)
    	x_train = x_train[:,:,np.newaxis]
    	x_test = x_test[:,:,np.newaxis]
    	# 将分类数据转换为独热编码
    	y_train = utils.to_categorical(y_train, 2)
    	y_test = utils.to_categorical(y_test, 2)
    	# print(len(y_train))
    	# num = len(y_train)
    	x = np.array(x)
    	x = x[:,:,np.newaxis]
    	# print(x)
    	y = utils.to_categorical(y, 2)
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


# from tensorflow.keras.utils import plot_model
def CreateDNNModel(Input_shape):
	model = Sequential()

	OPTIMIZER = Nadam(learning_rate = 0.002, beta_1 = 0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

	# 为模型添加层
	model.add(Flatten())
	model.add(Dense(200, input_shape=Input_shape, kernel_initializer=glorot_uniform(seed=0)))
	model.add(BatchNormalization())
	model.add(ReLU())
	model.add(Dropout(0.5))
	model.add(Dense(200, kernel_initializer=glorot_uniform(seed=0)))
	model.add(BatchNormalization())
	model.add(ReLU())
	model.add(Dropout(0.5))
	model.add(Dense(200, kernel_initializer=glorot_uniform(seed=0)))
	model.add(BatchNormalization())
	model.add(ReLU())
	model.add(Dropout(0.5))
	model.add(Dense(2, kernel_initializer=glorot_uniform(seed=0)))
	model.add(Activation('softmax'))
	# model.add(Activation('softmax'))

	model.compile(loss = 'categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
	# plot_model(model, to_file='./model.png')
	# model.summary()
	return model 

from Arithmetic.DNN_Attention import Attention, Average, WeightedSum, WeightedAspectEmb, MaxMargin
from Arithmetic.DNN_Attention import AttLayer
def create_weighted_model(num_path, num_all):
	matrix_input = Input(shape = (num_path, num_all, ), name = 'matrix_input')
	print(matrix_input)
	# 调用注意力机制
	# y_s = Average(num_path)(matrix_input)
	# att_weights = Attention(name='att_weights')([matrix_input, y_s])
	# z_s = WeightedSum()([matrix_input, att_weights])

	z_s = AttLayer(7, 1)(matrix_input)
	print(z_s)
	# 全连接学习向量特征
	p_t = Dense(500)(z_s)
	p_t = Activation('relu', name='p_t')(p_t)
	p_t = Dropout(0.5)(p_t)

	p_t1 = Dense(500)(p_t)
	p_t1 = Activation('relu', name='p_t1')(p_t1)
	p_t1 = Dropout(0.5)(p_t1)

	out = Dense(2)(p_t1)
	out = Activation('softmax', name = 'out')(out)

	model = Model(inputs = matrix_input, outputs = out)

	return model

def split_train_data(x_train, y_train, rate):
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 1-rate, random_state = 12)
    x_train = np.array(x_train)
    # print(type(x_train))
    # print(x_train.shape)
    x_test = np.array(x_test)
    # print(x_test)
    # x_train = x_train[:,:,:,np.newaxis]
    # x_test = x_test[:,:,:,np.newaxis]
    # 将分类数据转换为独热编码
    # y_train = utils.to_categorical(y_train, 2)
    # y_test = utils.to_categorical(y_test, 2)

    return x_train, y_train

def train_different_rate():
	ker_AIIPPIIATrans, labels = test_AIIPPIIA()
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

	x_train, x_test, y_train, y_test, x, y = Get_data(test_new, labels)

	# 对测试数据集进行分割，分别训练10%的数据，20%，30%-100%的数据，说明其稳定性
	rate = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	for i in rate:
		with open('test_split.txt', 'a') as f:
			f.write('使用训练集大小为' + str(i) + '\n')
		x_train, y_train = split_train_data(x_train, y_train, i)

		Input_shape = (ker_AATrans.shape[0], 1)
		model = CreateDNNModel(Input_shape)
	
		model.fit(x_train, y_train, epochs=20, batch_size = 128)
		# model.summary()
	
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
		with open('test_split.txt', 'a') as f:
			f.write('ACC:' + str(acc) + '\n')
			f.write('f1:' + str(2 * precision * recall / (precision + recall)) + '\n')
			f.write('tn: ' + str(benign_True) + '  |  fp: ' + str(malware_False) + '  |  fn: ' + str(benign_False) + '  |  tp: ' + str(malware_True) + '\n')

def train_single_path():
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


	x_train, x_test, y_train, y_test, x, y = Get_data(test_new, labels)
	
	Input_shape = (ker_AATrans.shape[0], 1)
	model = CreateDNNModel(Input_shape)

	model.fit(x_train, y_train, epochs=5, batch_size = 128)
	model.summary()

	# model.save("../data/Model/attention_5.h5")

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
	'''
	with open('DNNtrain.txt', 'a') as f:
		for i in range(len(y_test)):
			f.write(str(y_pred[i][0]) + ' ' + str(y_pred[i][1]) + ' ' + str(y_test[i][0]) + ' ' + str(y_test[i][1]) + '\n')
	'''
	# 绘制ROC曲线
	# test_y_score = fitted.decision_function(x_test)
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

def train_all_path():
	'''
	a1 = np.hstack((ker_AIIPPIIA, ker_AATrans))
	a2 = np.hstack((a1, ker_QQTrans))
	a3 = np.hstack((a2, ker_HHTrans))
	a4 = np.hstack((a3, ker_APPA))
	a5 = np.hstack((a4, ker_APPIIPPA))
	a6 = np.hstack((a5, ker_AIIA))
	'''
	# test_QA = ker_AATrans + ker_QQTrans
	# test_QA = np.hstack((ker_AATrans, ker_QQTrans))
	'''
	print(ker_AATrans.shape)
	print(ker_QQTrans.shape)
	print(ker_HHTrans.shape)
	print(ker_APPA.shape)
	print(ker_AIIA.shape)
	print(ker_AIIPPIIA.shape)
	print(ker_APPIIPPA.shape)
	'''

	ker_AATrans, ker_QQTrans, ker_HHTrans, ker_APPA, ker_AIIA, ker_APPIIPPA, ker_AIIPPIIA, labels = test_con()
	# 构造新的矩阵输入格式
	new_matrix = []
	
	# print(type(ker_AATrans))
	for i in range(ker_AATrans.shape[0]):
		# new_matrix_test = np.array(ker_AATrans[i], ker_QQTrans[i], ker_HHTrans[i], ker_APPA, ker_AIIA, ker_AIIPPIIA, ker_APPIIPPA)
		new_matrix.append([])
		new_matrix[i].append(ker_AATrans[i])
		new_matrix[i].append(ker_QQTrans[i])
		new_matrix[i].append(ker_HHTrans[i])
		new_matrix[i].append(ker_APPA[i])
		new_matrix[i].append(ker_AIIA[i])
		new_matrix[i].append(ker_AIIPPIIA[i])
		new_matrix[i].append(ker_APPIIPPA[i])
		new_matrix[i] = np.array(new_matrix[i])
	
	# print(new_matrix)
	new_matrix = np.array(new_matrix)
	# print(type(new_matrix))
	# print(new_matrix.shape)
	

	# new_matrix = np.array(new_matrix)
	x_train, x_test, y_train, y_test, x, y = Get_data(new_matrix, labels, k = 1)
	# print(x_train[0][0])

	# print(type(x_train))
	# print(x_train.shape[0],x_train.shape[1], x_train.shape[2])
	# print(x_train[0][0])
	num_path = 7

	OPTIMIZER = Nadam(learning_rate = 0.002, beta_1 = 0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

	model = create_weighted_model(num_path, ker_AATrans.shape[0])
	model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])

	# model.fit(x_train, y_train, epochs=10, batch_size = 256)
	model.fit(x_train, y_train, epochs=10, batch_size = 128)
	model.summary()

	# model.save("../data/Model/attention_1007.h5")

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

if __name__ == '__main__':
	train_single_path()
	# train_all_path()
	# train_different_rate()








