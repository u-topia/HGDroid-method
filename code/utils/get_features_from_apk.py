import codecs
import os
import re
from smalisca.core.smalisca_module import ModuleBase
from smalisca.core.smalisca_logging import log
import sys
from tqdm import tqdm
import csv
import numpy as np

from xml.dom import minidom


class SmaliParser(ModuleBase):
    """Iterate through files and extract data
    Attributes:
        location (str): Path of dumped APK
        suffix (str): File name suffix
        current_path (str): Will be updated during parsing
        classes (list): Found classes
    """
    def __init__(self, location, suffix):
        self.location = location
        self.suffix = suffix
        self.current_path = None
        self.classes = []
        # print(self.location)

    def run(self):
        """Start main task"""
        self.parse_location()

    def parse_file(self, filename):
        """Parse specific file
        This will parse specified file for:
            * classes
            * class properties
            * class methods
            * calls between methods
        Args:
            filename (str): Filename of file to be parsed
        """
        with codecs.open(filename, 'r', encoding='utf8') as f:
            current_class = None
            current_method = None
            current_call_index = 0

            # Read line by line
            for l in f.readlines():
                if '.class' in l:
                    match_class = self.is_class(l)
                    if match_class:
                        current_class = self.extract_class(match_class)
                        # self.write2file(filewrite, current_class)
                        # for key in current_class:
                            # print(key, current_class[key])
                        self.classes.append(current_class)

                elif '.method' in l:
                    match_class_method = self.is_class_method(l)
                    if match_class_method:
                        m = self.extract_class_method(match_class_method)
                        # for key in m:
                            # print(key, m[key])
                        current_method = m
                        current_call_index = 0
                        current_class['methods'].append(m)

                elif 'invoke' in l:
                    match_method_call = self.is_method_call(l)
                    if match_method_call:
                        m = self.extract_method_call(match_method_call)

                        # Add calling method (src)
                        m['src'] = current_method['name']

                        # Add call index
                        m['index'] = current_call_index
                        current_call_index += 1

                        # Add call to current method
                        current_method['calls'].append(m)
                '''
                elif '.super' in l:
                    match_class_parent = self.is_class_parent(l)
                    if match_class_parent:
                        current_class['parent'] = match_class_parent

                elif '.field' in l:
                    match_class_property = self.is_class_property(l)
                    if match_class_property:
                        p = self.extract_class_property(match_class_property)
                        current_class['properties'].append(p)

                elif 'const-string' in l:
                    match_const_string = self.is_const_string(l)
                    if match_const_string:
                        c = self.extract_const_string(match_const_string)
                        current_class['const-strings'].append(c)
                '''

                
            '''
            print('current_class:')
            for key in current_class:
                print(key, current_class[key])
            print('current_method:')
            for key in current_method:
                print(key, current_method[key])
            print('current_call_index:' + str(current_call_index))
            '''
        # Close fd
        f.close()

    def parse_location(self):
        """Parse files in specified location"""
        for root, dirs, files in os.walk(self.location):
            for f in files:
                # print(f)
                if f.endswith(self.suffix):
                    # TODO: What about Windows paths?
                    file_path = root + "/" + f
                    # print(1)
                    # print(file_path)

                    # Set current path
                    self.current_path = file_path

                    # Parse file
                    # log.debug("Parsing file:\t %s" % f)
                    # print("Parsing file:\t %s" % f)
                    self.parse_file(file_path)

    def is_class(self, line):
        """Check if line contains a class definition
        Args:
            line (str): Text line to be checked
        Returns:
            bool: True if line contains class information, otherwise False
        """
        match = re.search("\.class\s+(?P<class>.*);", line)
        if match:
            # log.debug("Found class: %s" % match.group('class'))
            # print("Found class: %s" % match.group('class'))
            return match.group('class')
        else:
            return None

    def is_class_parent(self, line):
        """Check if line contains a class parent definition
        Args:
            line (str): Text line to be checked
        Returns:
            bool: True if line contains class parent information, otherwise False
        """
        match = re.search("\.super\s+(?P<parent>.*);", line)
        if match:
            # log.debug("\t\tFound parent class: %s" % match.group('parent'))
            # print("\t\tFound parent class: %s" % match.group('parent'))
            return match.group('parent')
        else:
            return None

    def is_class_property(self, line):
        """Check if line contains a field definition
        Args:
            line (str): Text line to be checked
        Returns:
            bool: True if line contains class property information,
                  otherwise False
        """
        match = re.search("\.field\s+(?P<property>.*);", line)
        if match:
            #log.debug("\t\tFound property: %s" % match.group('property'))
            # print("\t\tFound property: %s" % match.group('property'))
            return match.group('property')
        else:
            return None

    def is_const_string(self, line):
        """Check if line contains a const-string
        Args:
            line (str): Text line to be checked
        Returns:
            bool: True if line contains const-string information,
                  otherwise False
        """
        match = re.search("const-string\s+(?P<const>.*)", line)
        if match:
            # log.debug("\t\tFound const-string: %s" % match.group('const'))
            # print("\t\tFound const-string: %s" % match.group('const'))
            return match.group('const')
        else:
            return None

    def is_class_method(self, line):
        """Check if line contains a method definition
        Args:
            line (str): Text line to be checked
        Returns:
            bool: True if line contains method information, otherwise False
        """
        match = re.search("\.method\s+(?P<method>.*)$", line)
        if match:
            # log.debug("\t\tFound method: %s" % match.group('method'))
            # print("\t\tFound method: %s" % match.group('method'))
            return match.group('method')
        else:
            return None

    def is_method_call(self, line):
        """Check [MaÔif the line contains a method call (invoke-*)
        Args:
            line (str): Text line to be checked
        Returns:
            bool: True if line contains call information, otherwise False
        """
        match = re.search("invoke-\w+(?P<invoke>.*)", line)
        if match:
            # log.debug("\t\t Found invoke: %s" % match.group('invoke'))
            # print("\t\t Found invoke: %s" % match.group('invoke'))
            return match.group('invoke')
        else:
            return None

    def extract_class(self, data):
        """Extract class information
        Args:
            data (str): Data would be sth like: public static Lcom/a/b/c
        Returns:
            dict: Returns a class object, otherwise None
        """
        class_info = data.split(" ")
        # log.debug("class_info: %s" % class_info[-1].split('/')[:-1])
        # print("class_info: %s" % class_info[-1].split('/')[:-1])
        c = {
            # Last element is the class name
            'name': class_info[-1],

            # Package name
            'package': ".".join(class_info[-1].split('/')[:-1]),

            # Class deepth
            'depth': len(class_info[-1].split("/")),

            # All elements refer to the type of class
            'type': " ".join(class_info[:-1]),

            # Current file path
            'path': self.current_path,

            # Properties
            'properties': [],

            # Const strings
            'const-strings': [],

            # Methods
            'methods': []
        }

        return c

    def extract_class_property(self, data):
        """Extract class property info
        Args:
            data (str): Data would be sth like: private cacheSize:I
        Returns:
            dict: Returns a property object, otherwise None
        """
        prop_info = data.split(" ")

        # A field/property is usually saved in this form
        #  <name>:<type>
        prop_name_split = prop_info[-1].split(':')

        p = {
            # Property name
            'name': prop_name_split[0],

            # Property type
            'type': prop_name_split[1] if len(prop_name_split) > 1 else '',

            # Additional info (e.g. public static etc.)
            'info': " ".join(prop_info[:-1])
        }

        return p

    def extract_const_string(self, data):
        """Extract const string info
        Args:
            data (str): Data would be sth like: v0, "this is a string"
        Returns:
            dict: Returns a property object, otherwise None
        """
        match = re.search('(?P<var>.*),\s+"(?P<value>.*)"', data)

        if match:
            # A const string is usually saved in this form
            #  <variable name>,<value>

            c = {
                # Variable
                'name': match.group('var'),

                # Value of string
                'value': match.group('value')
            }

            return c
        else:
            return None

    def extract_class_method(self, data):
        """Extract class method info
        Args:
            data (str): Data would be sth like:
                public abstract isTrue(ILjava/lang/..;ILJava/string;)I
        Returns:
            dict: Returns a method object, otherwise None
        """
        method_info = data.split(" ")

        # A method looks like:
        #  <name>(<arguments>)<return value>
        m_name = method_info[-1]
        m_args = None
        m_ret = None

        # Search for name, arguments and return value
        match = re.search(
            "(?P<name>.*)\((?P<args>.*)\)(?P<return>.*)", method_info[-1])

        if match:
            m_name = match.group('name')
            m_args = match.group('args')
            m_ret = match.group('return')

        m = {
            # Method name
            'name': m_name,

            # Arguments
            'args': m_args,

            # Return value
            'return': m_ret,

            # Additional info such as public static etc.
            'type': " ".join(method_info[:-1]),

            # Calls
            'calls': []
        }

        return m

    def extract_method_call(self, data):
        """Extract method call information
        Args:
            data (str): Data would be sth like:
            {v0}, Ljava/lang/String;->valueOf(Ljava/lang/Object;)Ljava/lang/String;
        Returns:
            dict: Returns a call object, otherwise None
        """
        # Default values
        c_dst_class = data
        c_dst_method = None
        c_local_args = None
        c_dst_args = None
        c_ret = None

        # The call looks like this
        #  <destination class>) -> <method>(args)<return value>
        match = re.search(
            '(?P<local_args>\{.*\}),\s+(?P<dst_class>.*);->' +
            '(?P<dst_method>.*)\((?P<dst_args>.*)\)(?P<return>.*)', data)

        if match:
            c_dst_class = match.group('dst_class')
            c_dst_method = match.group('dst_method')
            c_dst_args = match.group('dst_args')
            c_local_args = match.group('local_args')
            c_ret = match.group('return')

        c = {
            # Destination class
            'to_class': c_dst_class,

            # Destination method
            'to_method': c_dst_method,

            # Local arguments
            'local_args': c_local_args,

            # Destination arguments
            'dst_args': c_dst_args,

            # Return value
            'return': c_ret
        }

        return c

    def get_results(self):
        """Get found classes in specified location
        Returns:
            list: Return list of found classes
        """
        return self.classes

                    

class XMLParser(ModuleBase):
    '''
    该类用于分析XML文件，从中提取akp相对应的权限信息和硬件信息
    '''
    RequestedPermissionSet = set()
    ActivitySet = set()
    ServiceSet = set()
    ContentProviderSet = set()
    BroadcastReceiverSet = set()
    HardwareComponentsSet = set()
    IntentFilterSet = set()

    def __init__(self, filename):
        self.filename = filename
        return

    def parse_file(self):
        with open(self.filename, 'r') as f:
            Dom = minidom.parse(f)
            DomCollection = Dom.documentElement

            DomPermission = DomCollection.getElementsByTagName("uses-permission")
            for Permission in DomPermission:
                if Permission.hasAttribute("android:name"):
                    self.RequestedPermissionSet.add(Permission.getAttribute("android:name"))

            '''
            DomActivity = DomCollection.getElementsByTagName("activity")
            for Activity in DomActivity:
                if Activity.hasAttribute("android:name"):
                    self.ActivitySet.add(Activity.getAttribute("android:name"))

            DomService = DomCollection.getElementsByTagName("service")
            for Service in DomService:
                if Service.hasAttribute("android:name"):
                    self.ServiceSet.add(Service.getAttribute("android:name"))

            DomContentProvider = DomCollection.getElementsByTagName("provider")
            for Provider in DomContentProvider:
                if Provider.hasAttribute("android:name"):
                    self.ContentProviderSet.add(Provider.getAttribute("android:name"))

            DomBroadcastReceiver = DomCollection.getElementsByTagName("receiver")
            for Receiver in DomBroadcastReceiver:
                if Receiver.hasAttribute("android:name"):
                    self.BroadcastReceiverSet.add(Receiver.getAttribute("android:name"))
            '''

            DomHardwareComponent = DomCollection.getElementsByTagName("uses-feature")
            for HardwareComponent in DomHardwareComponent:
                if HardwareComponent.hasAttribute("android:name"):
                    self.HardwareComponentsSet.add(HardwareComponent.getAttribute("android:name"))
            '''
            DomIntentFilter = DomCollection.getElementsByTagName("intent-filter")
            DomIntentFilterAction = DomCollection.getElementsByTagName("action")
            for Action in DomIntentFilterAction:
                if Action.hasAttribute("android:name"):
                    self.IntentFilterSet.add(Action.getAttribute("android:name"))
            '''

    def run(self):
        self.parse_file()
        return self.HardwareComponentsSet, self.RequestedPermissionSet



def get_all_results(API):
    nameall = []
    '''
    with open('../data/already_apktool.txt', 'r') as f:
        for i in f.readlines():
            i = i[:-1]
            nameall.append(i)
    print(nameall[0])
    '''
    smali_path = '../data/family_test'
    drebin_list = os.listdir(smali_path)
    print(drebin_list)
    # print(drebin_list)
    # 遍历所有的该路径下的子目录
    # every_class = []
    file_need = ['a']
    for dl in drebin_list:
        if dl not in file_need:
            continue
        # 获取所有文件名称包括drebin的子集
        in_dl_path = os.path.join(smali_path, dl)
        # print(in_dl_path)
        in_dl_p = os.listdir(in_dl_path)
        # 获取所有drebin下的数据具体apktool反编译后的文件信息
        for idl in tqdm(in_dl_p):
            if idl in nameall:
                continue
            inin_dl_path = os.path.join(in_dl_path, idl)
            # 获取所有反汇编之后获得的文件路径
            # inin_dl_p = os.listdir(inin_dl_path)
            # with open('../data/apktool_after/files.txt', 'a') as f:
            #    f.write(inin_dl_path + ' malware\n')
            # 进入具体路径，分析其中的smali及xml文件信息
            '''
            for iidl in inin_dl_p:
                key_path = os.path.join(inin_dl_path, iidl)
                print(key_path)
                Smali = SmaliParser(iidl, '.smali')
                Smali.run()
                classes_all = Smali.get_results()
                return classes_all 
            '''
            print(inin_dl_path)
            Smali = SmaliParser(inin_dl_path, '.smali')
            Smali.run()
            classes_all = Smali.get_results()
            get_API_message(idl, classes_all, API)
            # every_class.append(classes_all)
            # return every_class         

def get_API_from_file():
    API = []
    file_path = '../data/API/entities.txt'
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for i in reader:
            if i[1] == '2':
                API.append(i[0])
    # print(API)
    return API


def get_API_message(idl, classes_all, API):
    # 提取APK中所有被调用的方法，包括自己定义的以及API信息
    method = []
    # API_number = np.zeros(len(API))
    for i in range(len(classes_all)):
        for j in classes_all[i]['methods']:
            for x in j['calls']:
                class1 = x['to_class']
                method_1 = x['to_method']
                if method_1 == '<init>':
                    method_1 = 'init'
                method_all_name = str(class1.replace('/', '.')) + '.' + str(method_1)
                method_all_name = method_all_name[1:]
                if method_all_name in API and method_all_name not in method:
                    method.append(method_all_name)
                # print(method_all_name)
    with open('../data/matrix/APK_API_familytest.txt', 'a') as f:
        f.write(idl + ' ' + str(len(method)) + '\n')
        for i in range(len(method)):
            f.write(method[i] + '\n')

def get_xml_message():
    hardwares = []
    permissions = []
    # path_all = '../data/apktool_after_benign/bangongxuexi/2haopeixun/AndroidManifest.xml'
    path_all = '../data/'
    # file = ['apktool_after_benign/', 'apktool_after_malware/']
    # file1 = ['drebin-0', 'drebin-1', 'drebin-2', 'drebin-3', 'drebin-4', 'drebin-5', 'bangongxuexi', 'jinronglicai']
    # file = ['AndroZoo/']
    # file1 = ['benign', 'malware']
    file = ['family_test']
    file1 = ['a']
    for file_first in file:
        path_all1 = path_all + file_first
        path_all_list = os.listdir(path_all1)
        for pal in path_all_list:
            if pal not in file1:
                continue
            in_path_all = os.path.join(path_all1, pal)
            in_path_all_list = os.listdir(in_path_all)
            for ipal in tqdm(in_path_all_list):
                in_in_path_all = os.path.join(in_path_all, ipal)
                try:
                    in_in_path_all += '/AndroidManifest.xml'
                    XMLFile = XMLParser(in_in_path_all)
                    hardwares, permissions = XMLFile.run()
                    with open('../data/matrix/APK_Per_hard_familyTest.txt', 'a') as f:
                        f.write(ipal + '  appname\n')
                        f.write('hardwares ' + str(len(hardwares)) + '\n')
                        for i in hardwares:
                            f.write(i + '\n')
                        f.write('permissions ' + str(len(permissions)) + '\n')
                        for i in permissions:
                            f.write(i + '\n')
                except:
                    print('file' + in_in_path_all + ' parses error!')

    '''
    XMLFile = XMLParser(path_all)
    hardwares, permissions = XMLFile.run()
    print(hardwares)
    print(permissions)
    '''

import time

if __name__=='__main__':
    time_start = time.time()
    # 获取API官方文档中所有的API信息
    # API = get_API_from_file()
    # get_all_results(API)
    get_xml_message()
    time_end = time.time()
    print("总用时为" + str(time_end - time_start))
    