import utils.get_relation_from_API
import utils.get_features_from_apk
import csv
from scipy import sparse
import pandas as pd


# 为每个API,package,permission创建唯一索引值
def getIDofAPIandPackage():
	package_num = 0
	API_num = 0
	permission_num = 0
	package = {}
	API = {}
	permission = {}
	with open('../data/API/entities.txt', 'r') as f:
		reader = csv.reader(f)
		for row in reader:
			if row[1] == '1':
				package[row[0]] = package_num
				package_num += 1
			if row[1] == '2':
				API[row[0]] = API_num
				API_num += 1
			if row[1] == '3':
				permission[row[0]] = permission_num
				permission_num += 1
	return API, package, permission

# 构建API与package之间的关系矩阵
def CreateAPIPac_Matrix(uniqueIDAPI, uniqueIDPackage):
	APIPac_matrixRows = []
	APIPac_matrixCols = []
	data = []
	with open('../data/matrix/API_pac.txt', 'r') as f:
		devide = True
		i = 0
		me_data = []
		for row in f.readlines():
			row = row[:-1]
			if devide:
				me_data = row.split(' ')
				# print(me_data)
				devide = False
				if me_data[1] == '0':
					devide = True
				continue
			if not devide:
				APIPac_matrixRows.append(uniqueIDAPI[row])
				APIPac_matrixCols.append(uniqueIDPackage[me_data[0]])
				data.append(1)
				i += 1
				if i == int(me_data[1]):
					devide = True 
					i = 0
	return sparse.coo_matrix((data, (APIPac_matrixRows, APIPac_matrixCols)))

# 为每个APP建立唯一索引
def getIDofAPP(filename):
	uniqueIDAPP = {}
	APP = 0
	labels = {}
	with open(filename, 'r') as f:
		for i in f.readlines():
			i = i[:-1]
			data = i.split(' ')
			if data[0][-4:] == '.apk':
				data[0] = data[0][:-4]
			uniqueIDAPP[data[0]] = APP 
			labels[data[0] + data[1]] = APP
			APP += 1
			# if APP >= 300:
			# 	break
	# print(labels)
	# print(len(labels))
	# print(len(uniqueIDAPP))
	return uniqueIDAPP, labels

def CreateAPPAPI_Matrix(uniqueIDAPP, uniqueIDAPI, filename):
	APPAPI_matrixRows = []
	APPAPI_matrixCols = []
	data = []
	with open(filename, 'r') as f:
		devide = True
		i = 0
		me_data = []
		for row in f.readlines():
			row = row[:-1]
			if devide:
				me_data = row.split(' ')
				devide = False
				if me_data[1] == '0':
					devide = True
				continue
			if not devide:
				if me_data[0] in uniqueIDAPP.keys():
					APPAPI_matrixRows.append(uniqueIDAPP[me_data[0]])
					APPAPI_matrixCols.append(uniqueIDAPI[row])
					data.append(1)
				i += 1
				if i == int(me_data[1]):
					devide = True 
					i = 0
	return sparse.coo_matrix((data, (APPAPI_matrixRows, APPAPI_matrixCols)))


def CreateAPIPer_Matrix(uniqueIDAPI, uniqueIDPermission):
	APIPer_matrixRows = []
	APIPer_matrixCols = []
	data = []
	with open('../data/matrix/API_Per.txt', 'r') as f:
		for row in f.readlines():
			row = row[:-1]
			de_data = row.split(' ')
			if '<init>' in de_data[0]:
				de_data[0] = de_data[0].replace('<init>', 'init')
			APIPer_matrixRows.append(uniqueIDAPI[de_data[0]])
			APIPer_matrixCols.append(uniqueIDPermission[de_data[2]])
			data.append(1)
	return sparse.coo_matrix((data, (APIPer_matrixRows, APIPer_matrixCols)))

def CreateAPPHard_APPPer_Matrix(uniqueIDAPP, uniqueIDPermission, hardwares, filename):
	APPHard_matrixRows = []
	APPHard_matrixCols = []
	data1 = []
	APPPer_matrixRows = []
	APPPer_matrixCols = []
	data2 = []
	# hardwares = {}
	hardwares_num = len(hardwares)
	with open(filename, 'r') as f:
		# devide1表示对于硬件的行是否拆分，devide2表示对权限的行是否拆分
		# i表示对硬件的计数，j表示对权限的计数
		devide1 = True
		devide2 = True
		i = 0
		j = 0
		de_data = []
		de_data_hardware = []
		de_data_permission = []
		# h = 0
		for row in f.readlines():
			# print(h)
			# h += 1
			row = row[:-1]
			if 'appname' in row:
				de_data = row.split(' ')
			if devide1 and 'hardwares' in row:
				devide1 = False
				de_data_hardware = row.split(' ')
				if de_data_hardware[1] == '0':
					devide1 = True
				continue
			if devide2 and 'permissions' in row :
				de_data_permission = row.split(' ')
				devide2 = False
				if de_data_permission[1] == '0':
					devide2 = True
				continue
			if not devide1:
				a = 0
				if row not in hardwares.keys():
					if hardwares_num < 65:
						hardwares[row] = hardwares_num
						hardwares_num += 1
						a = 1
				else:
					a = 1
				if a == 1:
					if de_data[0] in uniqueIDAPP.keys():
						APPHard_matrixRows.append(uniqueIDAPP[de_data[0]])
						APPHard_matrixCols.append(hardwares[row])
						data1.append(1)
				i += 1
				if i == int(de_data_hardware[1]):
					devide1 = True 
					i = 0
			if not devide2:
				if row not in uniqueIDPermission.keys():
					# print('非官方权限\n')
					j += 1
					continue
				if de_data[0] in uniqueIDAPP.keys():
					APPPer_matrixRows.append(uniqueIDAPP[de_data[0]])
					APPPer_matrixCols.append(uniqueIDPermission[row])
					data2.append(1)
				j += 1
				if j == int(de_data_permission[1]):
					devide2 = True
					j = 0
	if len(APPPer_matrixCols) != 270:
		for i in range(270):
			if i not in APPPer_matrixCols:
				APPPer_matrixRows.append(0)
				APPPer_matrixCols.append(i)
				data2.append(0)
	
	if len(APPHard_matrixCols) != 65:
		for i in range(65):
			if i not in APPHard_matrixCols:
				APPHard_matrixRows.append(0)
				APPHard_matrixCols.append(i)
				data1.append(0)
	return hardwares, sparse.coo_matrix((data1, (APPHard_matrixRows, APPHard_matrixCols))), sparse.coo_matrix((data2, (APPPer_matrixRows, APPPer_matrixCols)))


if __name__ == '__main__':
	# save_path = '../data/matrix/'
	uniqueIDAPI, uniqueIDPackage, uniqueIDPermission = getIDofAPIandPackage()
	# matrixAPIPac = CreateAPIPac_Matrix(uniqueIDAPI, uniqueIDPackage)
	# print(matrixAPIPac)
	uniqueIDAPP, labels = getIDofAPP('../data/label.txt')
	# print(len(uniqueIDAPP))
	# matrixAPPAPI = CreateAPPAPI_Matrix(uniqueIDAPP, uniqueIDAPI, '../data/matrix/APK_API.txt')
	# print(matrixAPPAPI)
	# matrixAPIPer = CreateAPIPer_Matrix(uniqueIDAPI, uniqueIDPermission)
	# print(matrixAPIPer)
	uniqueIDHard, matrixAPPHard, matrixAPPPer = CreateAPPHard_APPPer_Matrix(uniqueIDAPP, uniqueIDPermission, '../data/matrix/APK_Per_hard.txt')
	# print(matrixAPPHard)
	print(uniqueIDHard)
	

