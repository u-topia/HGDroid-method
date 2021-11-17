import sys
sys.path.append('../utils/')
import get_HIN_graph
from logs import Logger

from tensorflow.keras import utils
from keras.models import Sequential
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adamax, Nadam
from tensorflow import keras

from sklearn.model_selection import train_test_split

import numpy as np 
import tensorflow as tf
import pandas as pd

def get_data(ker, labels):
    df = pd.DataFrame(ker)
    # print(df)
    # print(len(labels))
    # print(len(df))
    df['classification_id'] = labels.keys()
    df['classification_id'] = df['classification_id'].apply(lambda x: 1 if 'malware' in x else 0)

    feature_cols = df.iloc[:, :-1]
    x = np.array(feature_cols)
    y = df['classification_id']
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 12)

    x = x[:,:,np.newaxis]
    # 将分类数据转换为独热编码
    # y_train = utils.to_categorical(y_train, 2)
    # y_test = utils.to_categorical(y_test, 2)
    y = utils.to_categorical(y, 2)

    # print("训练数据数量:", x_train.shape[0])
    # print("测试数据数量:", x_test.shape[0])
    # return x_train, x_test, y_train, y_test
    return x, y

def test_attention():
	model = keras.models.load_model('../data/Model/attention_5.h5')

	uniqueIDAPP_test, labels_test = get_HIN_graph.getIDofAPP('../data/label_family.txt')
	uniqueIDAPP_train, labels_train = get_HIN_graph.getIDofAPP('../data/label.txt')

	with open('family_malware1.txt', 'a') as f:
		for key in uniqueIDAPP_test:
			f.write(key + ' ' + str(uniqueIDAPP_test[key]) + '\n')

	# 获取矩阵数据
	uniqueIDAPI, uniqueIDPackage, uniqueIDPermission = get_HIN_graph.getIDofAPIandPackage()

	# 获得API与package关系矩阵，以P简称表示
	# matrixAPIPac = get_HIN_graph.CreateAPIPac_Matrix(uniqueIDAPI, uniqueIDPackage)

	# 获得APP与API关系矩阵，以A简称表示
	matrixAPPAPI_test = get_HIN_graph.CreateAPPAPI_Matrix(uniqueIDAPP_test, uniqueIDAPI, '../data/matrix/APK_API_familytest.txt')
	matrixAPPAPI_train = get_HIN_graph.CreateAPPAPI_Matrix(uniqueIDAPP_train, uniqueIDAPI, '../data/matrix/APK_API.txt')

	# 获得API与Permission之间的关系（PScout），以I简称表示
	# matrixAPIPer = get_HIN_graph.CreateAPIPer_Matrix(uniqueIDAPI, uniqueIDPermission)

	# 获得APP与Hardwares之间的关系矩阵，以H简称表示
	# 获得APP与Permission之间的关系矩阵，以Q简称表示
	uniqueIDHard = {}
	uniqueIDHard, matrixAPPHard_train, matrixAPPPer_train = get_HIN_graph.CreateAPPHard_APPPer_Matrix(uniqueIDAPP_train, uniqueIDPermission, uniqueIDHard, '../data/matrix/APK_Per_hard.txt')
	uniqueIDHard, matrixAPPHard_test, matrixAPPPer_test = get_HIN_graph.CreateAPPHard_APPPer_Matrix(uniqueIDAPP_test, uniqueIDPermission, uniqueIDHard,'../data/matrix/APK_Per_hard_familyTest.txt')

	AATrains = matrixAPPAPI_test.dot(matrixAPPAPI_train.transpose())

	ker_AATrains = AATrains.toarray()

	HHTrains = matrixAPPHard_test.dot(matrixAPPHard_train.transpose())

	ker_HHTrains = HHTrains.toarray()
	
	test_new = []
	for i in range(ker_AATrains.shape[0]):
		test_new.append([])
		test_new[i] = ker_AATrains[i] * 0.5 + ker_HHTrains[i] * 0.5
	test_new = np.array(test_new)
	
	x_test, y_test = get_data(test_new, labels_test)
	
	# x_test, y_test = get_data(ker_AATrains, labels_test)
	score_test = model.evaluate(x_test, y_test, verbose=2)
	print('测试集精度为：', score_test[1])
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
	with open('family_malware1.txt', 'a') as f:
		for i in range(len(y_test)):
			f.write(str(y_pred[i][0]) + ' ' + str(y_pred[i][1]) + '\n')


if __name__ == '__main__':
	test_attention()