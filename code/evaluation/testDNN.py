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

def test_DNNAAT():
	model = keras.models.load_model('../data/Model/AATrans_0930.h5')

	uniqueIDAPP_test, labels_test = get_HIN_graph.getIDofAPP('../data/AndroZoo/labels.txt')
	uniqueIDAPP_train, labels_train = get_HIN_graph.getIDofAPP('../data/label.txt')

	# 获取矩阵数据
	uniqueIDAPI, uniqueIDPackage, uniqueIDPermission = get_HIN_graph.getIDofAPIandPackage()

	# 获得API与package关系矩阵，以P简称表示
	# matrixAPIPac = get_HIN_graph.CreateAPIPac_Matrix(uniqueIDAPI, uniqueIDPackage)

	# 获得APP与API关系矩阵，以A简称表示
	matrixAPPAPI_test = get_HIN_graph.CreateAPPAPI_Matrix(uniqueIDAPP_test, uniqueIDAPI, '../data/matrix/APK_API_test.txt')
	matrixAPPAPI_train = get_HIN_graph.CreateAPPAPI_Matrix(uniqueIDAPP_train, uniqueIDAPI, '../data/matrix/APK_API.txt')

	# 获得API与Permission之间的关系（PScout），以I简称表示
	# matrixAPIPer = get_HIN_graph.CreateAPIPer_Matrix(uniqueIDAPI, uniqueIDPermission)

	# 获得APP与Hardwares之间的关系矩阵，以H简称表示
	# 获得APP与Permission之间的关系矩阵，以Q简称表示
	# uniqueIDHard, matrixAPPHard, matrixAPPPer = get_HIN_graph.CreateAPPHard_APPPer_Matrix(uniqueIDAPP, uniqueIDPermission, '../data/matrix/APK_Per_hard_test.txt')

	AATrains = matrixAPPAPI_test.dot(matrixAPPAPI_train.transpose())

	ker_AATrans = AATrains.toarray()

	x_test, y_test = get_data(ker_AATrans, labels_test)

	score_test = model.evaluate(x_test, y_test, verbose=2)
	print('测试集精度为：', score_test[1])

def test_DNNQQT_HHT():
	# model = keras.models.load_model('../data/Model/DNN_QQT.h5')

	model1 = keras.models.load_model('../data/Model/DNN_HHT.h5')

	uniqueIDAPP_test, labels_test = get_HIN_graph.getIDofAPP('../data/AndroZoo/labels.txt')
	uniqueIDAPP_train, labels_train = get_HIN_graph.getIDofAPP('../data/label.txt')

	# 获取矩阵数据
	uniqueIDAPI, uniqueIDPackage, uniqueIDPermission = get_HIN_graph.getIDofAPIandPackage()

	# 获得API与package关系矩阵，以P简称表示
	# matrixAPIPac = get_HIN_graph.CreateAPIPac_Matrix(uniqueIDAPI, uniqueIDPackage) 

	# 获得APP与Hardwares之间的关系矩阵，以H简称表示
	# 获得APP与Permission之间的关系矩阵，以Q简称表示
	uniqueIDHard = {}
	uniqueIDHard, matrixAPPHard_train, matrixAPPPer_train = get_HIN_graph.CreateAPPHard_APPPer_Matrix(uniqueIDAPP_train, uniqueIDPermission, uniqueIDHard, '../data/matrix/APK_Per_hard.txt')
	uniqueIDHard, matrixAPPHard_test, matrixAPPPer_test = get_HIN_graph.CreateAPPHard_APPPer_Matrix(uniqueIDAPP_test, uniqueIDPermission, uniqueIDHard, '../data/matrix/APK_Per_hard_test.txt')

	print(str(matrixAPPHard_train.shape[0]) + ' ' + str(matrixAPPHard_train.shape[1]))
	print(str(matrixAPPHard_test.shape[0]) + ' ' + str(matrixAPPHard_test.shape[1]))
	# QQTrains = matrixAPPPer_test.dot(matrixAPPPer_train.transpose())

	HHTrains = matrixAPPHard_test.dot(matrixAPPHard_train.transpose())

	# ker_QQTrans = QQTrains.toarray()
	ker_HHTrains = HHTrains.toarray()

	# x_test, y_test = get_data(ker_QQTrans, labels_test)
	x_test, y_test = get_data(ker_HHTrains, labels_test)

	score_test = model1.evaluate(x_test, y_test, verbose=2)
	print('测试集精度为：', score_test[1])

def test_APPA():
	model = keras.models.load_model('../data/Model/DNN_APPA.h5')

if __name__ == '__main__':
	# test_DNNAAT()
	test_DNNAAT()
