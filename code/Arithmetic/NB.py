import sys
sys.path.append('../')
import utils.get_HIN_graph as get_HIN_graph
from utils.logs import Logger
import pandas as pd
import numpy as np

from sklearn import svm 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

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

def train_NB(ker, labels):
	df = pd.DataFrame(ker)
	# print(df)
	# print(len(labels))
	# print(len(df))
	df['classification_id'] = labels.keys()
	df['classification_id'] = df['classification_id'].apply(lambda x: 1 if 'malware' in x else -1)

	feature_cols = df.iloc[:, :-1]
	x = feature_cols
	y = df['classification_id']

	# Logger('LinearSVC').get_log().debug('元路径为AIIPPIIA，test_size = 0.25, 包括5560恶意APP以及1119良性APP')
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 12)
	
	# 构建朴素贝叶斯学习网络
	clf = GaussianNB()
	# clf.fit(x_train, y_train)
	fitted = clf.fit(x_train, y_train)
	pred = fitted.predict(x_test)
	clf.score(x_test, y_test)
	accuracy = fitted.score(x_test, y_test)
	print('----------------------------------------------------------')
	print('Accuracy: ' + str(accuracy))
	print('----------------------------------------------------------')
	# Logger('linearSVC').get_log().debug('Accuracy: ' + str(accuracy))
	f1 = f1_score(y_test, pred, average = 'weighted', labels = np.unique(pred))
	print('F1_score: ' + str(f1))
	# Logger('linearSVC').get_log().debug('F1_score: ' + str(f1))
	tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
	print('Confusion_matrix')
	print('----------------------------------------------------------')
	print('tn: ' + str(tn) + '  |  fp: ' + str(fp) + '  |  fn: ' + str(fn) + '  |  tp: ' + str(tp))

	# 绘制ROC曲线
	# test_y_score = fitted.decision_function(x_test)
	fpr, tpr, threshold = roc_curve(y_test, pred)
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
	ker_AATrans = AATrans.toarray()
	train_NB(ker_AATrans, labels)

def test_QQT():
	# 对QQ^T进行分类测试，表示每个APP所需求的权限对分类的影响
	P, A, I, H, Q, labels = get_all_matrix()
	QQTrans = getAATrans(Q, trans(Q))
	ker_QQTrans = QQTrans.toarray()
	train_NB(ker_QQTrans, labels)

def test_HHT():
	# 对HH^T进行分类测试，表示APP调用了相同的权限信息
	P, A, I, H, Q, labels = get_all_matrix()
	HHTrans = getAATrans(H, trans(H))
	ker_HHTrans = HHTrans.toarray()
	train_NB(ker_HHTrans, labels)

def test_APPA():
	# 对路径APPA进行分类测试，表示两个APP调用的API属于同一个包
	P, A, I, H, Q, labels = get_all_matrix()
	AP = A.dot(P)
	APP = AP.dot(trans(P))
	APPA = APP.dot(trans(A))
	ker_APPA = APPA.toarray()
	train_NB(ker_APPA, labels)

def test_APPIIPPA():
	# 路径APPIIPPA表示调用的API中属于同一个包的具有相同的权限
	P, A, I, H, Q, labels = get_all_matrix()
	APPIIPPA = A.dot(P).dot(trans(P)).dot(I).dot(trans(I)).dot(P).dot(trans(P)).dot(trans(A))
	ker_APPIIPPA = APPIIPPA.toarray()
	train_NB(ker_APPIIPPA, labels)

def test_AIIPPIIA():
	# 路径AIIPPIIA表示与调用的API具有相同权限的API属于同一个包
	P, A, I, H, Q, labels = get_all_matrix()
	AIIPPIIA = A.dot(I).dot(trans(I)).dot(P).dot(trans(P)).dot(I).dot(trans(I)).dot(trans(A))
	ker_AIIPPIIA = AIIPPIIA.toarray()
	train_NB(ker_AIIPPIIA, labels)

def test_con():
	P, A, I, H, Q, labels = get_all_matrix()
	AATrans = getAATrans(A, trans(A))
	ker_AATrans = AATrans.toarray()
	HHTrans = getAATrans(H, trans(H))
	ker_HHTrans = HHTrans.toarray()
	test_new = []
	for i in range(ker_AATrans.shape[0]):
		test_new.append([])
		test_new[i] = ker_AATrans[i] * 0.5 + ker_HHTrans[i] * 0.5
	test_new = np.array(test_new)
	train_NB(test_new, labels)

# 利用SVM进行分类测试
def main():
	# test_AAT()
	# test_QQT()
	# test_HHT()
	# test_APPA()
	# test_APPIIPPA()
	# test_AIIPPIIA()
	test_con()
			

if __name__ == '__main__':
	main()