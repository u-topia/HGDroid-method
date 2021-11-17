import os
import csv


# 保存APP名及其对应的标签信息
def getLabelofAPP():
	path = '../data/'
	path_next = ['apktool_after_benign', 'apktool_after_malware']
	file1 = ['drebin-0', 'drebin-1', 'drebin-2', 'drebin-3', 'drebin-4', 'drebin-5', 'bangongxuexi', 'jinronglicai']
	for pn in path_next:
		path_new = path + pn 
		path_list = os.listdir(path_new)
		for pl in path_list:
			if pl not in file1:
				continue
			in_path = os.path.join(path_new, pl)
			in_path_list = os.listdir(in_path)
			with open('../data/label.txt', 'a') as f:
				if 'benign' in in_path:
					for i in in_path_list:
						f.write(i + ' benign\n')
				else:
					for i in in_path_list:
						f.write(i + ' malware\n')
			
# 将提取到的矩阵信息写入到文件中保存
def write2file(filename, matrix):
	with open(filename, 'w') as f:
		writer = csv.writer(f)
		writer.writerows(matrix)


if __name__ == '__main__':
	getLabelofAPP()