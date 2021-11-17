import os
from tqdm import tqdm
import json 
import csv
import re

# 从API官方文档中获取所有的API方法以及相应的关系
# 包括属于同一个包，具有相同的权限，see-alse关系等信息
def get_all_API():
	doc_path = '../data/API/API_docs_in_json'
	json_files = os.listdir(doc_path)
	global entity_type
	entity_type = {
	'package': 1,
	'method': 2,
	'permission': 3
	}
	all_entities = {}
	for jf in tqdm(json_files):
		jf_path = os.path.join(doc_path, jf)
		# 将获取到的信息保存到all_entities这个字典中
		get_Entities(jf_path, all_entities)
	loadPermissionsExternal(all_entities)
	loadEntitiesInExternal(all_entities)
	save_entities = [[entity, all_entities[entity]] for entity in all_entities]
	save_entities.sort(key = lambda x:x[1])
	with open('../data/API/entities.txt', 'w', newline = '') as f:
		writer = csv.writer(f)
		writer.writerows(save_entities)
	return save_entities

# 获得json文件中有关API方法以及所属包的信息
def get_Entities(json_path, all_entities):
	if not os.path.exists(json_path):
		return
	data = json.load(open(json_path))
	class_name = clean_entity_name(data['ClassName'])
	package_name = get_package_name_from_class(class_name)
	if package_name:
		all_entities[package_name] = entity_type['package']
	for method in data['Functions']:
		method_name = class_name + '.' + method[0:method.find('(')]
		all_entities[method_name] = entity_type['method']

# 对类名进行清洗，获得其最终结果
def clean_entity_name(s):
	re_entity = re.compile(r'@B_\S+_E@')
	entities = re_entity.findall(s)
	if entities:
		entity = s[3:-3].replace('#','.')
		return entity[:] if entity[0] != '.' else entity[1:]
	else:
		result = s 
		if '<' in result:
			k = result.find('<')
			result = result[0:k]
		if '(' in result:
			k = result.find('(')
			result = result[0:k]
		result = result.replace('[]', '')
		if '#' in result:
			k1 = result.find('#')
			result = result[k1 + 1:]
		return result

# 获取所有的权限信息
def loadPermissionsExternal(all_entities):
	per_path = '../data/API/all_permissions.txt'
	with open(per_path, 'r') as f:
		for line in f:
			permission = line.strip()  # 移除首位的空格
			if permission:
				all_entities[permission] = entity_type['permission']
	return

# 获取非官方定义的方法信息
def loadEntitiesInExternal(all_entities):
	ex_path = '../data/API/extra_permission_relations.txt'
	with open(ex_path, 'r') as f:
		for line in f:
			temp = line.strip().split(' ')
			method = clean_method(temp[0])
			all_entities[method] = entity_type['method']

# 从类名中获取包名
def get_package_name_from_class(class_name):
	for i, c in enumerate(class_name):
		if c.isupper():
			return class_name[0:i-1]

# 对方法名进行清洗
def clean_method(method):
    method = method.replace('.<init>', '.init')
    return method

def get_API_from_same_package():
	map_API_pac = {}
	with open('../data/API/entities.txt', 'r') as f:
		reader = csv.reader(f)
		for row in reader:
			if row[1] == '1':
				map_API_pac[row[0]] = []
			if row[1] == '2':
				package_name = get_package_name_from_class(row[0])
				# print(package_name)
				# map_API_pac[package_name] = []
				try:
					map_API_pac[package_name].append(row[0])
				except:
					continue
			if row[1] == '3':
				break
	# print(len(map_API_pac))
	return map_API_pac


def main():
	# get_all_API()
	map_API_pac = get_API_from_same_package()
	with open('../data/matrix/API_pac.txt', 'w') as f:
		for key, data in map_API_pac.items():
			f.write(key + ' ' + str(len(data)) + '\n')
			for i in range(len(data)):
				f.write(data[i] + '\n')

if __name__ == '__main__':
	main()